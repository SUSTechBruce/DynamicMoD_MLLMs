from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

# cd /users/PAS2473/brucewan666/ACL2025_Dynamiclayer_MOD/Dynamic_MoD/dynamic_mod
# bash  ./scripts/train/finetune_eval_7b_pmod_llava_next.sh
