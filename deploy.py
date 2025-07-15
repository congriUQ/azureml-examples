from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command
from dotenv import load_dotenv
import os

load_dotenv()

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential,
    subscription_id=os.getenv("subscription_id"),
    resource_group_name="sales-ai",
    workspace_name="sales-ai-aml-workspace"
)

print(ml_client)

job = command(
    code="./",  # local folder
    command="python hello_world.py",
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",  # prebuilt environment
    compute="A1v2",
    experiment_name="my-experiment",
    display_name="my-script-run"
)

# returned_job = ml_client.jobs.create_or_update(job)
# print(f"Job submitted: {returned_job.name}")


