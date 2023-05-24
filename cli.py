import engine

print('AIPA LLM CLI 1.0.0 (Experiment, May 21 2023, 14:16)')
print('Type "help" for more information, "exit() to exit the CLI."')

cache = []

while True:
    user_query = str(input(">>> "))
    if "exit()" in user_query:
        exit()
    
    question = ''

    for prev_ques in cache:
        question = question + ' ' + prev_ques
    
    question = question + ' \n ' + 'Now question is: "' + user_query  + '"'

    # print(question)

    engine_response = engine.answer_user_query(question)
    print(engine_response, '\n')
    cache.append('Q: ' + user_query + '\n' + 'A: ' + str(engine_response))