from lmdeploy import pipeline

CKPT = '/home/ubuntu/uq/checkpoints/checkpoint-100'

if __name__ == '__main__':
    print(f'Loading {CKPT} via lmdeploy...')
    pipe = pipeline(CKPT)
    response = pipe('What is 2 + 2? Think step by step.')
    print('=== Response ===')
    print(response.text)
