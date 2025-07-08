from mii import pipeline
pipe = pipeline('/home/mihir/data/engines/Qwen2-7B')
output = pipe(["Hello, my name is", "DeepSpeed is"], max_new_tokens=128)
print(output)