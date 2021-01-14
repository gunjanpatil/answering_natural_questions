pip install -r requirements.txt
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ../
sudo pip install gsutil
mkdir datasets
cd datasets
mkdir natural_questions_simplified
cd natural_questions_simplified
gsutil -m cp -R gs://natural_questions/v1.0-simp* .
cd v1.0-simplified/
gzip -d simplified-nq-train.jsonl.gz
gzip -d nq-dev-all.jsonl.gz
