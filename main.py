from bot import QaBot

if __name__ == '__main__':

    bot = QaBot(model='chatglm', doc_path="data/《中华人民共和国民法典》.txt", url='http://127.0.0.1:8000')
    questions = [
        "自己在院中种的树,是否有权自行砍伐?"
        ]
    for question in questions:
        print("问题：")
        print(question)
        answer = bot.ask(question)
        print("回答: ")
        print(answer)

