import requests
import time


def get_glm45_response(messages, temperature=0.0, tools=None):
    start = time.time()
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": "Bearer 94accb09d3fd4dbbbc2459e79eede036.X67OUpQParjVXldD",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": "glm-4.5-air", # glm-4.5-air
        "messages": messages,
        "thinking": {
            "type": "enabled"
        },
        "tools": tools,
        "max_tokens": 40000,
    }
    proxies={
        "http": "http://10.1.4.213:3128",
        "https": "http://10.1.4.213:3128",
    }
    try:
        response = requests.post(url, json=json_data, headers=headers, proxies=proxies, timeout=300)
        # print(response.text)
        response = response.json()
        print("cost time: {}".format(time.time() - start))
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print("exception", e)
        time.sleep(300)
        return None



def test():
    prompt = "2025年8月4日（北京时间）发生了什么大事，有哪些重大新闻？"
    messages = [{"role": "user", "content": prompt}]
    tools = ["web_search"]
    res = get_glm45_response(messages, 0, tools)
    print("res", res)


if __name__ == "__main__":
    test()