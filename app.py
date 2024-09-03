import os
import json
import nest_asyncio
import threading
from openai import OpenAI
from flask import Flask, request, jsonify, render_template
import argparse
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pymongo.mongo_client import MongoClient
from simple_salesforce import Salesforce


openapi_key = os.getenv('OPENAI_API_KEY')
mongo_uri = os.getenv('MONGO_URI')
sf_username = os.getenv('SF_USERNAME')
sf_password = os.getenv('SF_PASSWORD')
sf_security_token = os.getenv('SF_SECURITY_TOKEN')

nest_asyncio.apply()

app = Flask(__name__)


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    client = OpenAI(
        api_key=openapi_key,
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        #temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

def create_vector_search():
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        mongo_uri,
        namespace="UML_ChatBot.demo-db",
        embedding= OpenAIEmbeddings(),
        index_name="vector_index"
    )
    return vector_search

messages = []

q=[]
c=[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    def main(query_text):
        flag1235 = 0
        chat_example = '''
        [
            {"role": "system", "content": "You are a helpful assistant designed to provide information related to the University of Massachusetts, Lowell."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "{\n\"response\": \"Hello! How can I assist you today?\",\n\"code\": \"1234\"\n}"},
            {"role": "user", "content": "pantry?"},
            {"role": "assistant", "content": "{\n\"response\": \"The main Strive Pantry is located at the Graduate Professional Studies Center! Visitors to the pantry should stop by the information desk on the first floor, where they will be greeted by staff and escorted to the pantry.\",\n\"code\": \"1234\"\n}"},
            {"role": "user", "content": "timings?"},
            {"role": "assistant", "content": "{\n\"response\": \"The pantry will re-open for the summer on Wednesday, May 29. The summer hours are: Mondays: 11 a.m. – 4:30 p.m., Wednesdays: 11:30 a.m. – 2:30 p.m., Fridays: 1–4 p.m.\",\n\"code\": \"1234\"\n}"},
            {"role": "user", "content": "Capital of USA?"},
            {"role": "assistant", "content": "{\n\"response\": \"I'm here to provide information specifically related to the University of Massachusetts, Lowell. Unfortunately, I do not have information on general knowledge questions. Is there anything else I can assist you with regarding UMass Lowell?\",\n\"code\": \"4321\"\n}"},
            {"role": "user", "content": "I want to report issue"},
            {"role": "assistant", "content": "{\n\"response\": \"Sure, I can help with that. Please provide me with the following details in the mentioned format:  \
        `name`, `UID`, `description of issue`. Note: Order of details provided is important \",\n\"code\": \"1234\"\n}"},
            {"role": "user", "content": "details: ak, 23, Want to raise a request for change of my last name"},
            {"role": "assistant", "content": "{\n\"response\": \"{\n\"name\": \"ak\",\n\"UID\" : \"23\",\n\"description\": \"Want to raise a request for change of my last name\"\n}\",\n\"code\": \"1235\"\n}"}
        ]
        '''
        
        NOT_FOUND_RESPONSE = "I'm sorry, I don't have enough sources to answer that question. My role is to provide information related to University of Massachusetts, Lowell. Is there anything else I can help you with?"
        
        PROMPT_TEMPLATE = """
        You are a AI chatbot. Your role is to build conversation with student and give good responses.
        If you are not confident on any question. Let student know that you don't have that information.
        Use the below context to complete the conversation. Strictly look for answer only in context
        Try extracting maximum outcome from context
        Context is the text between `````` below
        context : ```{context}```
        
        To report an issue, request the student to provide information in the following format:
        ##`name`, `UID`, `description of issue`##
        For example:
        ##'Jane Smith, 789012, I’m unable to access my course materials on the portal.'##
        If any details are missing, request the additional information from the student. Above provided all 3 details are mandatory.
        If student provided all the details, then don't search for answer in context.
        
        Response formatting:
        If answer is taken from context or student stating issue or student provided only partial details attach code `1234` to every assistant message
        else IF student provided all the details as mentioned in above example delimited by ##, then the <answer> must be {answer_} and attach code '1235'
        else attach code `4321`
        Format your responses as a JSON object in the following format:
        {normal_response}
        
        Find example conversation below in between ###### as reference:
        ###{chat_example}###
        """
    
        answer_ = "{\n'name': `name`,\n'UID' : `UID`,\n'description': `description`\n}"
        
        normal_response = "{\n\"response\": <answer>,\n\"code\": <code>\n}"
        
        issue_response = "{\n\"response\": {\n\"name\": <name>,\n\"UID\": <UID>,\n\"description\": <description>\n},\n\"code\": \"1235\"\n}"
        
        vector_search = create_vector_search()
        
        model = ChatOpenAI()
        q.append(query_text)
        c[0]+=1
        if c[0]==5:
            q.pop(0)
            c[0]-=1
        
        q_text = ' '.join(q)
    
        results = vector_search.similarity_search_with_score(
            query=q_text,
            k=10,
        )
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        # if len(results) == 0 or results[0][1] < 0.65:
        #     context_text = ""
            
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, normal_response=normal_response, answer_=answer_, chat_example=chat_example)[8:]
        #print(prompt)
        if messages == []:
            messages.append({'role':'system', 'content':f"{prompt}"})
    
        messages[0]={'role':'system', 'content':f"{prompt}"}
        messages.append({'role':'user', 'content':f"{query_text}"})
    
        try:
            res1 = get_completion_from_messages(messages)
            res = json.loads(res1)
        
            if res['code']=="4321":
                #print('error....................')
                res['response'] = NOT_FOUND_RESPONSE
            if res['code']=="1235":
                flag1235 = 1
                sf = Salesforce(username=sf_username, password=sf_password, security_token=sf_security_token)
                print('hello123 ',res['response'])
                uid = res['response']['UID']
                query_contact= sf.query(f"SELECT Id from Contact where Student_Id__c = '{uid}'")
    
                if len(query_contact['records']) == 1:
                    contact_id = query_contact['records'][0]['Id']
                    print("Contact ID:", contact_id)
                    case_data = {
                    'Subject': res['response']['description'],
                    'Description': res['response']['description'],
                    'Priority': 'Medium',
                    'Status': 'New',
                    'ContactId': contact_id
                    }
                    result = sf.Case.create(case_data)
                    query_case = sf.query("SELECT Id, CaseNumber from Case")
                    return_ans = (f"Case created with ID: {result['id']} and Case Number: {query_case['records'][0]['CaseNumber']}")
                
                else:
                    messages.pop(-1)
                    return_ans = "Invalid Student ID"
                    return jsonify({'response': return_ans})

        except:
            messages.pop(-1)
            q.pop(-1)
            return main(query_text)
            # ---------------------------------------------------
            #print(res['response'])
            
    
        response = res['response']
        print('hi ',res)
        res = json.dumps(res)
        messages.append({'role':'assistant', 'content':f"{res}"})
        if flag1235 == 1:
            return jsonify({'response': return_ans})
        return jsonify({'response': response})
    
    data = request.json
    query_text = data.get('query')
    return main(query_text)


if __name__ == '__main__':
    app.run(debug=True)
