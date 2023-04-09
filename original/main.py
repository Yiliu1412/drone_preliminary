import airsim_client

if __name__ == "__main__":
    client = airsim_client.airsim_client('127.0.0.1')
    client.begin_task()
