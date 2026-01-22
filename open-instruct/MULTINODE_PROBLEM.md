I have the code open-instruct/run_grpo.slurm &  open-instruct/scripts/train/olmo3/qwen_math_1_5.sh ;
 It works well with the 1.5B on 2 GPUs. 
Now I want to scale it up to 7B ("Qwen/Qwen2.5-Math-7B"), the problem is that i need minimum 3GPUs to do the weights update. but in GRPO, the bottleneck is the rollouts / VLLM engine inference, not the weights updates, so i need multiple nodes, but I dont know to to do that. I am using snellius and i can get mutliple nodes with this command. 
I want to use salloc of 2node of 4 H100 gpus  for it. Ideally 3 gpus for weights update, 5 for inference. but if it's easier to implement , i am happy with 4 nodes for weights update, 4 for inference.

salloc --partition=gpu_h100 --gres=gpu:h100:2 --time=02:00:00 --nodes=2

But the code breaks with it from various reasons. 

The original code (run on another cluster) was this open-instruct/scripts/train/olmo3/7b_rlzero_math.sh . maybe it shows how to run correctly the training on multiple codes. 

srun --ntasks-per-node=1 bash salloc_setup.sh

GOAL: read the repo to understand precisly how to run the grpo training on 2 nodes, then write a documentation on it, and write a .sh file that i can run in the interactive session to start the training.

Be critical, if something seems off or task too dificult, it's better to stop and talk with me .

