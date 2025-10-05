#!/bin/bash
rsync -avz --exclude 'edu_fineweb10B/' -e "ssh -i ~/.ssh/id_ed25519" ~/Projects/gpt2/ ubuntu@104.171.203.189:~/gpt2/
ssh -t ubuntu@104.171.203.189
