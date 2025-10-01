#!/bin/bash
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" ~/Projects/gpt2/ ubuntu@192.222.57.212:~/gpt2/
ssh -t ubuntu@192.222.57.212 'cd ~/gpt2 && python3 train_gpt2.py'
