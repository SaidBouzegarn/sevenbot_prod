# src/modules/s3_utils.py

import boto3
import pandas as pd
import json
import io
import logging 
import os

logger = logging.getLogger(__name__)

# Replace with your OpenAI API key
BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

def create_s3_session():
    # Initialize the S3 client
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION,
    )
    return session.client("s3")


def read_csv_from_s3(bucket_name, key):
    """
    Read a CSV file from S3 and return it as a pandas DataFrame.
    """
    s3 = create_s3_session()
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="utf-8")
    except Exception as e:
        logger.error(f"Error reading CSV from S3: {e}")
        return None


def write_csv_to_s3(df, bucket_name, key):
    """
    Write a pandas DataFrame to a CSV file in S3.
    """
    s3 = create_s3_session()

    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=bucket_name, Key=key, Body=csv_buffer.getvalue())
    except Exception as e:
        logger.error(f"Error writing CSV to S3: {e}")


from botocore.exceptions import BotoCoreError, ClientError


def write_json_to_s3(data, bucket_name, key):
    """
    Write a Python dictionary (or list) as a JSON file to S3.

    Args:
        data (dict or list): The data to be written to S3.
        bucket_name (str): The name of the S3 bucket.
        key (str): The key (path/filename) for the S3 object.
        session (boto3.Session, optional): A boto3 session object. If not provided, will use the default session.

    Returns:
        bool: True if the operation succeeded, False otherwise.
    """

    s3 = create_s3_session()
    if not isinstance(bucket_name, str) or not isinstance(key, str):
        logger.info(f"Error writing CSV to S3: file key is {key}")
        raise ValueError("Both bucket_name and object_key must be strings.")
    
    # Validate bucket name
    if not bucket_name.islower() or not bucket_name.replace('-', '').isalnum():
        raise ValueError("Invalid bucket name. Ensure it is lowercase and contains only alphanumeric characters and hyphens.")


    body = json.dumps(data)

    # Write data to S3
    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    logger.info(f"Successfully wrote data to S3: s3://{bucket_name}/{key}")


def read_json_from_s3(bucket_name, key):
    """
    Read a JSON file from S3 and return it as a Python dictionary.
    """
    s3 = create_s3_session()

    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as e:
        logger.error(f"Error reading JSON from S3: {e}")
        return None


def check_file_exists_s3(bucket_name, key):
    """
    Check if a file exists in an S3 bucket.
    """
    s3 = create_s3_session()
    try:
        s3.head_object(Bucket=bucket_name, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def list_files_in_s3(bucket_name, prefix=""):
    """
    List all files in an S3 bucket with an optional prefix.
    """
    s3 = create_s3_session()

    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        return [item["Key"] for item in response.get("Contents", [])]
    except Exception as e:
        logger.error(f"Error listing files in S3: {e}")
        return []


def delete_file_from_s3(bucket_name, key):
    """
    Delete a file from an S3 bucket.
    """
    s3 = create_s3_session()

    try:
        s3.delete_object(Bucket=bucket_name, Key=key)
    except Exception as e:
        logger.error(f"Error deleting file from S3: {e}")
