# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from datetime import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datasets import load_dataset

import pandas as pd
from metrics import MultiTurnInstructionFollowingPromptSolution
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from vllm import LLM
from utils import GenerationSetting, get_inference_batch_vllm, preprocess_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main(
    model_path: str,
    revision: str,
    languages: list[str] = ["English", "Spanish"],
    batch_size=256,
    generation_setting=GenerationSetting(
        max_new_tokens=32768, temperature=0.6, top_p=0.95
    ),
    need_write2file: bool = False,
    output_dir: str = "data",
    output_filepath_prefix: str = "eval_result",
    tensor_parallel_size: int = 8,
    steps: List[int] = [1, 2, 3],
    system_prompt: str = "",
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = LLM(
        model_path,
        revision=revision,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        seed=generation_setting.seed,
    )

    org, model_name = model_path.split("/", 1)
    lighteval_metrics = {}
    lighteval_metrics["config_general"] = {
        "model_name": model_name,
        "model_sha": revision,
    }
    lighteval_metrics["results"] = {}

    for language in languages:
        logger.info(f"Evaluating language: {language}")
        dataset = load_dataset("HuggingFaceTB/Multi-IF", language, split="train[:2]")
        benchmark_df = dataset.to_pandas()
        num_rows = len(benchmark_df)
        logger.info(f"Number of rows: {num_rows}")

        final_metric_result = {}


        start = time.time()
        model_name = model_path.split('/')[-1].strip()
        step_input_df = benchmark_df.copy()

        for step in steps:
            output_filepath = (
                f"{output_dir}/results/{model_name}/{output_filepath_prefix}_step_{step}.csv"
            )
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            step_output_df = run_step(
                model=model,
                tokenizer=tokenizer,
                input_df=step_input_df,
                row_limit=-1,
                need_write2file=need_write2file,
                output_filepath=output_filepath,
                device=0,
                generation_setting=generation_setting,
                batch_size=batch_size,
                system_prompt=system_prompt if step == 1 else "",  # Only add system prompt for the first step
            )
            step_input_df = step_output_df.copy()
            step_metric_result = run_metric(
                output_filepath=output_filepath,
                step=step,
            )
            final_metric_result[step] = step_metric_result

        logger.info(
            f"Total time: {time.time() - start}\n Number of rows: {num_rows}, \n Number of processes: 1"
        )
        logger.info(f"Final metrics: {final_metric_result}")

        for step, metrics in final_metric_result.items():
            lighteval_metrics["results"][f"turn_{step}_all_languages"] = metrics[f"turn_{step}_all_languages_overall"]
            lighteval_metrics["results"][f"turn_{step}_{language}"] = metrics[f"turn_{step}_{language}_overall"]

    # Create organized results directory structure: results/{ORG}/{MODEL_NAME}/{TIMESTAMP}/
    timestamp = datetime.now().isoformat().replace(":", "-")
    results_dir = Path(output_dir) / "results" / org / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(lighteval_metrics, f, indent=2)

def run_step(
    model: LLM,  # Changed to vllm's LLM
    tokenizer: PreTrainedTokenizerBase,
    input_df: pd.DataFrame,
    prompt_columns: List[str] = ["turns", "responses"],
    step: int = 0,
    row_limit: int = -1,
    need_write2file: bool = True,
    output_filepath: str = "eval_result.csv",
    device: Optional[str] = None,
    generation_setting: GenerationSetting = GenerationSetting(),
    batch_size: int = 256,
    system_prompt: str = "",
) -> pd.DataFrame:
    output_df = preprocess_data(
        input_df, prompt_columns=prompt_columns, row_limit=row_limit
    )
    step_output_df = get_inference_batch_vllm(
        model=model,
        tokenizer=tokenizer,
        input_df=output_df,
        batch_size=batch_size,
        generation_setting=generation_setting,
        need_write2file=need_write2file,
        output_filepath=output_filepath,
        device=device,
        system_prompt=system_prompt,
    )
    return step_output_df


def run_metric(
    output_filepath: str = "eval_result", step: int = 0
) -> Dict[str, Any]:
    step_output_df = None
    csv = output_filepath
    logger.info(f"calculating metrics for step_{step}")
    temp_df = pd.read_csv(csv, keep_default_na=False)
    step_output_df = temp_df.copy()
    metric_result = MultiTurnInstructionFollowingPromptSolution.metrics_gen(
        step_output_df
    )
    metric_result_df = pd.DataFrame.from_dict(metric_result, orient="index")
    metric_result_df.to_csv(output_filepath.replace('.csv', '_metric.csv'), index=False)
    logger.info(f"step_{step} metrics \n: {metric_result}")
    return metric_result


if __name__ == "__main__":
    """
    !!NOTE!!: make sure the number of available GPUs == tensor_parallel_size
    Usage:
    python multi_turn_instruct_following_eval_vllm.py \
        --model_path <MODEL_PATH> \
        --revision <REVISION> \
        --language <LANGUAGE> \
        --batch_size <BATCH_SIZE> \
        --need_write2file <NEED_WRITE2FILE> \
        --output_filepath_prefix <OUTPUT_FILEPATH_PREFIX> \
        --tensor_parallel_size <TENSOR_PARALLEL_SIZE>

    Example:
    python multi_turn_instruct_following_eval_vllm.py \
        --model_path meta-llama/Llama-3.1-8B-Instruct \
        --revision main \
        --language Spanish \
        --batch_size 256 \
        --tensor_parallel_size 1

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Meta-Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--revision", type=str, default="main"
    )
    parser.add_argument("--languages", type=str, nargs='+', default=["English", "Spanish"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--need_write2file", type=bool, default=True)
    parser.add_argument("--output_filepath_prefix", type=str, default="eval_result")
    parser.add_argument(
        "--output_dir", type=str, default="data"
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=[1, 2, 3],
        help='List of steps to process (e.g., --steps 1 2 3)'
    )
    parser.add_argument(
        "--system_prompt", 
        type=str, 
        default="", 
        help="System prompt to add as the first message in conversations"
    )
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        revision=args.revision,
        languages=args.languages,
        batch_size=args.batch_size,
        need_write2file=args.need_write2file,
        output_dir=args.output_dir,
        output_filepath_prefix=args.output_filepath_prefix,
        tensor_parallel_size=args.tensor_parallel_size,
        steps=args.steps,
        system_prompt=args.system_prompt,
    )
