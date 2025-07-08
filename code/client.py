import mii

deployment_name = "Qwen/Qwen2-7B"

generator = mii.pipeline(deployment_name)

response = generator("Who is the president of the United States?", max_new_tokens=50)

print("Response:", response)
