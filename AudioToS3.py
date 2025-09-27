import boto3
import uuid

def UploadToS3(audio_bytes, bucket="agromareh", region="eu-north-1"):
    """
    Uploads audio bytes to a public S3 bucket and returns the public URL.
    """
    s3 = boto3.client("s3", region_name=region)

    # Generate random filename
    filename = f"{uuid.uuid4().hex}.mp3"

    # Upload the audio
    s3.put_object(
        Bucket=bucket,
        Key=filename,
        Body=audio_bytes,
        ContentType="audio/mpeg",
    )

    # Return direct public URL
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{filename}"
    return url
