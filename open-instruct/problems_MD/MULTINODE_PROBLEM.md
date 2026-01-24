I have the code open-instruct/run_grpo.slurm &   open-instruct/scripts/train/olmo3/qwen_math_1_5.sh ;
It works well on a single node on snellius . 

Now I want to scale it up to 7B ("Qwen/Qwen2.5-Math-7B"), and I need multiple nodes (at least 2).

I want to use for now 2node of 4 H100 gpus for it. One full node for inference (actor/vllm engines), one full node for weight update (learner). 

GOAL: read the repo to understand precisly how to run the grpo training on 2 nodes, and write a separete .slurm & .sh file for 7B file.

Be critical, if something seems off or task too dificult, it's better to stop and discuss with me!

If you need, look online and search for stuff,