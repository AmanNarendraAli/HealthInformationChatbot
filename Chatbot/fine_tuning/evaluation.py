from datetime import datetime
from dotenv import load_dotenv
from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip
from langsmith import Client
from langchain.smith import RunEvalConfig
from langchain_core.runnables import RunnableLambda
from med_assist.config import CONFIG, set_project_wd
from med_assist.chain import build_chain

set_project_wd()
load_dotenv()

# Prepare dataset 

data_path = download_and_unzip(
    url = CONFIG['beir']['url'],
    out_dir = CONFIG['beir']['path']
    )
corpus, queries, qrels = GenericDataLoader(
    data_folder = data_path
    ).load(split="test")

dataset_input = [{"question": question} for question in list(queries.values())[:100]]

# configure langsmith evaluation

client = Client()

dataset_name = "med_assist_evaluation_" + datetime.now().strftime('%m%d-%H%M')
dataset_descr = "Medical assistant evaluation"

if client.has_dataset(dataset_name=dataset_name):
    raise NameError("Dataset name already exist")

dataset = client.create_dataset(
    dataset_name=dataset_name,
    description=dataset_descr
)

client.create_examples(
    dataset_name=dataset_name, 
    inputs=dataset_input)

eval_config = RunEvalConfig(
    evaluators=[
        RunEvalConfig.Criteria("conciseness"),
        RunEvalConfig.Criteria("harmfulness")
        ]
)

chain = build_chain()
format_input = RunnableLambda(lambda d: d.get("question"))

eval_chain = format_input | chain

client.run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=eval_chain,
    # evaluation=eval_config,
    verbose=True,
    project_name="evaluation_test",
    concurrency_level=1
)