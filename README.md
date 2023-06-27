## 基于Chatglm/ChatGPT使用自定义嵌入层

自定义嵌入层,把文档嵌入到向量空间中,然后计算相似度,找到最相似的文档,作为回复

1. chatglm:
    
    先启动chatglm服务端
    ```shell
    $python3 api.py
    ```
   chatglm的api默认端口为8000(如果不是请修改响应代码)
    然后启动chatbot
    ```shell
    $python3 main.py
    ```
    bot = QaBot(model='chatglm', doc_path="data/《中华人民共和国民法典》.txt", url='http://127.0.0.1:8000')
2. chatgpt:
    需要修改ApiKey,然后启动chatbot
    bot = QaBot(model='chatgpt', doc_path="data/《中华人民共和国民法典》.txt", key='XXXXXXXXXXXX')

   
测试问题:

![20230627170316.png](assets%2F20230627170316.png)