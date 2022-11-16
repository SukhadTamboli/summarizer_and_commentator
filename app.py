import string
import sys
import pymongo
from flask import Flask, jsonify, redirect, render_template, request, json, session, send_from_directory
from flask_session import Session
import pandas as pd

from passlib.hash import sha256_crypt
from functools import wraps
import json
from flask_cors import CORS, cross_origin
import os
import subprocess

app = Flask(__name__)

cors = CORS(app, resources={
            r"/register": {"origins": "*"}}, static_folder='../frontend/build')
app.config.from_pyfile('config.py')
Session(app)

sess = Session()
sess.init_app(app)

frontend = 'http://commentator-iitgn.s3-website.ap-south-1.amazonaws.com'
# conn_str = os.environ.get("DATABASE_URL")
conn_str = "mongodb+srv://new_user_7:584nYp9q17XE5VrR@cluster77.yfqtoem.mongodb.net/?retryWrites=true&w=majority" 
#changed link and linked to our mongo db server
#Added CSV WITH EVERY LINE TEXT NO COLOUM HEADER

# set a 5-second connection timeout
client = pymongo.MongoClient(conn_str, serverSelectionTimeoutMS=5000)
database = client['annotation_tool']
try:
    print("\nConnected to the db.\n")
except Exception:
    print("Unable to connect to the server.")


@app.route('/test', methods=['GET'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def test():
    return jsonify({'result': 'Hello World'})


@app.route('/signup', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def register():
    user_collection = database.get_collection('users')
    requestdata = json.loads(request.data)
    requestdata = json.loads(requestdata['body'])
    # print(requestdata, type(requestdata))
    # print()
    # for elem in requestdata:
    #     print(elem)
    username = requestdata['username']
    password = sha256_crypt.encrypt(str(requestdata['password']))
    print(username, password)

    result = user_collection.find({'username': username})
    res = list(result)
    print('Res: ', res, '\tType: ', type(res))
    print(res.__len__())
    if res.__len__() > 0:
        return jsonify({"error_message": "The username has already been taken"})
    else:
        user_collection.insert_one(
            {'username': username, 'password': password, 'sentId': 0, 'admin': False, 'sentTag': []})

        result = {
            'username': username,
            'password': password,
            'message': "Your account has been created. Please Login!"
        }
        return jsonify({'result': result})
    # return jsonify({'result': requestdata})


@app.route('/login', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def login():
    user_collection = database.get_collection('users')

    requestdata = json.loads(request.data)
    requestdata = json.loads(requestdata['body'])

    username = requestdata['username']
    password = str(requestdata['password'])
    print(username, password)

    result = user_collection.find({'username': username})
    res = list(result)
    print('Res: ', res, '\tType: ', type(res))
    print(res.__len__())
    if res.__len__() > 0:
        data = res[0]
        print(data['password'])
        sentId = data['sentId']
        admin = data['admin'] if data['admin'] else False
        print(sentId)
        print(sha256_crypt.verify(password, data['password']))
        # userID = data['id']
        # role = data['role']

        if sha256_crypt.verify(password, data['password']):
            session['logged_in'] = True
            session['username'] = username
            # session['role'] = role
            # session['user_id'] = userID

            # return jsonify({ 'response': 'Login successful' })
        else:
            error = 'Invalid Password'
            return jsonify({'error': error})

    else:
        error = 'Username not found'
        return jsonify({'error': error})

    returning = {
        # 'userId': session['user_id'],
        'username': session['username'],
        'sentId': sentId,
        'admin': admin
    }
    return jsonify({'success': returning})


def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            return jsonify({'message': "You are not logged in!"})
    return wrap


@app.route('/logout', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
# @is_logged_in
def logout():
    session.clear()
    return jsonify({'message': "You are logged out"})


@app.route('/get-sentence', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type'])
def get_sentence():
    sentences_collection = database.get_collection('sentences')
    requestdata = json.loads(request.data)
    print(requestdata)
    requestdata = json.loads(requestdata['body'])

    sentId = requestdata['id']
    print(sentId)
    result = sentences_collection.find({'sid': sentId})
    data = list(result)
    data = data[0]
    sentence = data['sentence']

    # os.system("/LID_tool/getLanguage.py sampleinp.txt")

    result = {
        'sentence': sentence,
        'sentId': sentId,
        'message': "Sentence Fetched Successfully."
    }
    return jsonify({'result': result})


@app.route('/get-lid-data', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def lid_tag():
    # from LID_tool.getLanguage import langIdentify

    requestdata = json.loads(request.data)
    print(requestdata)
    requestdata = json.loads(requestdata['body'])

    sid = requestdata['sentId']
    print('SENTENCE = ', sid)

    # lang = langIdentify(sentence, 'classifiers/HiEn.classifier')
    # tags = []
    # print(lang)
    # for elem in lang:
    #     inter = [elem[0]]
    #     for i in range(1, len(elem)):
    #         if elem[i] is '1':
    #             inter.append(elem[i-1][0])
    #     if len(inter) == 1:
    #         inter.append('u')
    #     tags.append(inter)

    # print('LANGUAGE TAG = ', tags)
    lid_collection = database.get_collection('lid')
    prev = lid_collection.find()
    prev = list(prev)
    print(prev)
    tags = prev[int(sid)-1]['tags']
    return jsonify({'result': tags})


@app.route('/admin-file-upload', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def admin_file_upload():
    # requestdata = json.loads(request.data)
    # print(requestdata)
    print(request.files['file'])
    file = request.files['file']
    file.save('uploads/{}'.format(file.filename))
    # requestdata = json.loads(requestdata['body'])

    # file = requestdata['file']
    # print('FILE = ', file)
    import pandas as pd
    # os.system('db.py 1 {}'.format(file.filename))
    sentences_collection = database.get_collection('sentences')

    filename = file.filename
    df = pd.read_csv('uploads/{}'.format(filename), header=None)
    df = df.iloc[:, 0]
    print(df)

    last_row_id = 0
    print(last_row_id)
    prev = sentences_collection.find()
    prev = list(prev)
    if len(prev) > 0:
        prev = prev[-1]
        print(prev['sid'])
        last_row_id = prev['sid']

    for sent in range(len(df)):
        last_row_id += 1

        print(df[sent])
        sentences_collection.insert_one({
            'sentence': df[sent],
            'sid': last_row_id
        })

    print('Task Finished')

    # os.system('LID_execute.py 1 {}'.format(file.filename))
    from LID_tool.getLanguage import langIdentify
    lid_collection = database.get_collection('lid')

    sentences_collection = database.get_collection('sentences')
    prev_sent = sentences_collection.find()
    prev_sent = list(prev_sent)
    total_num_of_sent = len(prev_sent)

    last_row_id = 0
    print(last_row_id)
    prev = lid_collection.find()
    prev = list(prev)
    if len(prev) > 0:
        prev = prev[-1]
        print(prev['tag_id'])
        last_row_id = prev['tag_id']

    sentence_details = prev_sent[last_row_id]
    sentence = sentence_details['sentence']
    start_index = sentence_details['sid']
    print('SENTENCE = ', sentence)
    print(total_num_of_sent)
    print(prev_sent[start_index-1])

    for i in range(start_index-1, total_num_of_sent):
        sentence = prev_sent[i]['sentence']
        lang = langIdentify(sentence, 'classifiers/HiEn.classifier')
        tags = []

        print(lang)
        for elem in lang:
            inter = [elem[0]]
            for i in range(1, len(elem)):
                if elem[i] is '1':
                    inter.append(elem[i-1][0])
            if len(inter) == 1:
                inter.append('u')
            tags.append(inter)

        print('LANGUAGE TAG = ', tags)
        lid_collection.insert_one({
            'tags': tags,
            'tag_id': last_row_id + 1
        })
        last_row_id = last_row_id + 1

    return redirect('{}/admin'.format(frontend))


@app.route('/sentence-schema-creation', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def sentence_schema_creation():
    try:
        database.create_collection('users')
    except:
        print("Already exists")

    try:
        database.create_collection('sentences')
    except:
        print("Already exists")

    try:
        database.create_collection('lid')
    except:
        print("Already exists")

    print('Schemas Created')
    return redirect('{}/admin'.format(frontend))


@app.route('/fetch-users-list', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def fetch_users_list():
    user_collection = database.get_collection('users')
    user_list = user_collection.find({})
    user_list = list(user_list)

    users_list = []
    for user in user_list:
        users_list.append(user['username'])
    print(users_list)
    # user_list = list(user_collection)
    # print(user_list)

    return jsonify({'result': users_list})


@app.route('/csv-download', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def csv_download():
    from flask import send_file

    username = request.form.get('username')
    cmi = request.form.get('cmi')
    # os.system('db_to_csv.py {}'.format(username))
    import csv
    users_collection = database.get_collection('users')

    print('username = ', username)
    if(username != 'ALL'):
        user = users_collection.find({'username': username})
        user = list(user)
        print(user)
        sentTag = user[0]['sentTag']

        with open('csv/{}.csv'.format(username), 'w', encoding='utf-8', newline="") as f:
            writer = csv.writer(f)

            writer.writerow(['grammar', 'date', 'tag', 'link',
                            'hashtag', 'time', 'CMI Score'])

            for sentence in sentTag:
                # print(sentence)
                grammar = sentence[0]
                date = sentence[1]
                tag = sentence[2]
                link = sentence[3]
                hashtag = sentence[4] if sentence[4] else []
                time = sentence[5]
                row = [grammar, date, tag, link, hashtag, time]

                en_count = 0
                hi_count = 0
                token_count = 0
                lang_ind_count = 0

                for i in range(len(tag)):
                    if(tag[i]['value'] == 'e'):
                        en_count += 1
                    elif(tag[i]['value'] == 'h'):
                        hi_count += 1
                    elif(tag[i]['value'] == 'u'):
                        lang_ind_count += 1
                    token_count += 1

                max_w = max(en_count, hi_count)

                cmi_score = 0
                if(token_count > lang_ind_count):
                    cmi_score = 100 * \
                        (1 - (max_w / (token_count - lang_ind_count)))
                else:
                    cmi_score = 0

                if(cmi_score >= float(cmi)):
                    row.append(cmi_score)
                    writer.writerow(row)
                # break

        return send_file('csv/{}.csv'.format(username), as_attachment=True)
    else:
        user = users_collection.find()
        user = list(user)
        print(username['username'] for username in user)
        with open('csv/all.csv', 'w', encoding='utf-8', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['User', 'grammar', 'date', 'tag', 'link',
                                     'hashtag', 'time', 'CMI Score'])

            for single_user in user:
                sentTag = single_user['sentTag']
                for sentence in sentTag:
                    # print(sentence)
                    grammar = sentence[0]
                    date = sentence[1]
                    tag = sentence[2]
                    link = sentence[3]
                    hashtag = sentence[4] if sentence[4] else []
                    time = sentence[5]
                    row = [single_user['username'], grammar, date,
                           tag, link, hashtag, time]

                    en_count = 0
                    hi_count = 0
                    token_count = 0
                    lang_ind_count = 0

                    for i in range(len(tag)):
                        if(tag[i]['value'] == 'e'):
                            en_count += 1
                        elif(tag[i]['value'] == 'h'):
                            hi_count += 1
                        elif(tag[i]['value'] == 'u'):
                            lang_ind_count += 1
                        token_count += 1

                    max_w = max(en_count, hi_count)

                    cmi_score = 0
                    if(token_count > lang_ind_count):
                        cmi_score = 100 * \
                            (1 - (max_w / (token_count - lang_ind_count)))
                    else:
                        cmi_score = 0

                    if(cmi_score >= float(cmi)):
                        row.append(cmi_score)
                        if(single_user['admin'] is False):
                            writer.writerow(row)
                    # break

        return send_file('csv/all.csv', as_attachment=True)

    # print(username)
    # return jsonify({'result': 'Done'})

    # return redirect('{}/admin'.format(frontend))
    return


@app.route('/compare-annotators', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def compare_annotators():
    from flask import send_file

    username1 = request.form.get('username1')
    username2 = request.form.get('username2')
    kappa = request.form.get('kappa')
    print(username1, username2, kappa)

    # return jsonify({'result': 'true'})
    # os.system('compare.py {} {}'.format(username1, username2))
    import csv
    username1_name = username1
    username2_name = username2
    print('username1 = ', username1_name)
    print('username2 = ', username2_name)

    user_collection = database.get_collection('users')
    username1 = user_collection.find({'username': username1_name})
    username2 = user_collection.find({'username': username2_name})

    user1 = list(username1)
    user2 = list(username2)

    print('USER 1 = ', user1)
    print('USER 2 = ', user2)

    counter = min(int(user1[0]['sentId']), int(user2[0]['sentId']))
    print(counter)

    sentTag1 = user1[0]['sentTag']
    sentTag2 = user2[0]['sentTag']

    with open('csv/compare.csv', 'w', encoding='utf-8', newline="") as f:
        writer = csv.writer(f)

        writer.writerow(['grammar_{}'.format(username1_name), 'date_{}'.format(username1_name), 'tag_{}'.format(username1_name), 'link_{}'.format(username1_name), 'hashtag_{}'.format(username1_name), 'time_{}'.format(username1_name), '', 'grammar_{}'.format(username2_name), 'date_{}'.format(username2_name), 'tag_{}'.format(username2_name), 'link_{}'.format(username2_name), 'hashtag_{}'.format(username2_name),
                        'time_{}'.format(username2_name), '', 'grammer_same', 'words_with_similar_annotation', 'total_words', 'Cohen Kappa Score'])

        for count in reversed(range(counter)):
            # print(sentence)
            grammar_1 = sentTag1[count][0]
            date_1 = sentTag1[count][1]
            tag_1 = sentTag1[count][2]
            link_1 = sentTag1[count][3]
            hashtag_1 = sentTag1[count][4] if sentTag1[count][4] else []
            time_1 = sentTag1[count][5]

            empty = ''

            grammar_2 = sentTag2[count][0]
            date_2 = sentTag2[count][1]
            tag_2 = sentTag2[count][2]
            link_2 = sentTag2[count][3]
            hashtag_2 = sentTag2[count][4] if sentTag2[count][4] else []
            time_2 = sentTag2[count][5]

            grammer_same = 0
            if grammar_1 == grammar_2:
                grammer_same = 1

            words_with_similar_annotation = 0
            total_words = 0
            for index in range(len(tag_1)):
                if tag_1[index]['value'] == tag_2[index]['value']:
                    words_with_similar_annotation += 1
                total_words += 1

            from sklearn.metrics import cohen_kappa_score
            ann1_tags = [elem['value'] for elem in tag_1]
            ann2_tags = [elem['value'] for elem in tag_2]
            kappa_score = cohen_kappa_score(
                ann1_tags, ann2_tags, labels=None, weights=None)

            row = [grammar_1, date_1, tag_1, link_1, hashtag_1, time_1,
                   empty, grammar_2, date_2, tag_2, link_2, hashtag_2, time_2, empty, grammer_same, words_with_similar_annotation, total_words, kappa_score]

            print(kappa_score, type(kappa_score))
            if(float(str(kappa_score)) >= float(kappa)):
                writer.writerow(row)
            counter -= 1
            # break

    # return 'Good'
    return send_file('csv/compare.csv', as_attachment=True)


@app.route('/submit-sentence', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
# @is_logged_in
def submit_sentence():
    user_collection = database.get_collection('users')
    requestdata = json.loads(request.data)
    print(requestdata)
    requestdata = json.loads(requestdata['body'])
    print(requestdata)

    sentId = requestdata['sentId']
    selected = requestdata['selected']
    tag = requestdata['tag']
    username = requestdata['username']
    date = requestdata['date']
    hypertext = requestdata['hypertext']
    hashtags = requestdata['hashtags']
    timeDifference = requestdata['timeDifference']

    lst = [selected, date, tag, hypertext, hashtags, timeDifference]
    print(lst)

    print(sentId, selected, tag, username)

    user_collection.update_one({'username': username}, {
        '$set': {'sentId': sentId},
        '$push': {'sentTag': lst}
    })

    return jsonify({'result': 'Message Stored Successfully'})


# @app.route('/tokenize-en', methods=['POST'])
# # @is_logged_in
# def tokenize_en():
#     sentences_collection = database.get_collection('sentences')
#     requestdata = json.loads(request.data)
#     print(requestdata)
#     requestdata = json.loads(requestdata['body'])

#     sentId = requestdata['id']
#     print(sentId)
#     result = sentences_collection.find({'sid': sentId})
#     data = list(result)
#     data = data[0]
#     sentence = data['sentence']
#     print(sentence)

#     return jsonify({'result': 'Message Stored Successfully'})

@app.route('/get-edit-sentence', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
# @is_logged_in
def get_edit_sentence():
    user_collection = database.get_collection('users')
    requestdata = json.loads(request.data)
    print(requestdata)
    requestdata = json.loads(requestdata['body'])

    sentId = requestdata['id']
    username = requestdata['logged_in_user']

    user = user_collection.find({'username': username})
    user = list(user)
    user = user[0]
    userTags = user['sentTag'][sentId-1]

    # user_collection.update_one({'username': username}, {
    #     '$set': {'sentTag[{sentId}]'.format(sentId=sentId-1): lst},
    # })

    return jsonify({'result': userTags})


@app.route('/submit-edit-sentence', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
# @is_logged_in
def submit_edit_sentence():
    user_collection = database.get_collection('users')
    requestdata = json.loads(request.data)
    print(requestdata)
    requestdata = json.loads(requestdata['body'])

    sentId = requestdata['sentId']
    selected = requestdata['selected']
    tag = requestdata['tag']
    username = requestdata['username']
    date = requestdata['date']
    hypertext = requestdata['hypertext']
    hashtags = requestdata['hashtags']
    timeDifference = requestdata['timeDifference']

    lst = [selected, date, tag, hypertext, hashtags, timeDifference]

    print(lst)

    print(sentId, selected, tag, username)

    user = user_collection.find({'username': username})
    user = list(user)
    sentTag = user[0]['sentTag']
    sentTag[sentId - 1] = lst

    user_collection.update_one({'username': username}, {
        '$set': {'sentTag': sentTag}
    })

    # user_collection.update_one({'username': username}, {
    #     '$set': {'sentTag[{sentId}]'.format(sentId=sentId-1): lst},
    # })

    return jsonify({'result': 'Message Stored Successfully'})


@app.route('/all-sentences', methods=['GET', 'POST'])
# @is_logged_in
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def all_sentence():
    user_collection = database.get_collection('users')
    requestdata = json.loads(request.data)
    print(requestdata)
    requestdata = json.loads(requestdata['body'])

    username = json.loads(requestdata['username'])

    print('username: ', username)

    result = user_collection.find({'username': username})
    print(result)
    res = list(result)
    res = res[0]
    print(res)

    return jsonify({'result': res['sentTag']})

@app.route('/Summarizer', method =['GET', 'POST'])
def Summarizer():
    return{
        'userid': 1,
        'tittle': 'Flask running',
    }
stopwords = list(STOP_WORDS)
nlp = spacy.load('en_core_web_sm') #.bin 
doc = nlp(text)
tokens = [token.text for token in doc]
#print(tokens)
punctuation = punctuation + '\n'
word_frequencies = {}
for word in doc:
  if word.text.lower() not in stopwords:
        if word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1
max_frequency = max(word_frequencies.values())
for word in word_frequencies.keys():
  word_frequencies[word] = word_frequencies[word]/max_frequency
sentence_tokens = [sent for sent in doc.sents]
sentence_scores = {}
for sent in sentence_tokens:
  for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] += word_frequencies[word.text.lower()]
            select_length = int(len(sentence_tokens)*0.3)
summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
final_summary = [word.text for word in summary]
summary = ' '.join(final_summary)

print(summary)

''' Here we can also do this by:
def read_article(file_name):
    file = open(file_name,"r")
    filedata = file.readline
    article = filedata()[0].split(". ")
    sentences=[]
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]"," ").split(" "))
    # sentence.pop()
    sentence = sentence[:-1]
    return sentences
    # used for reading article

def sentence_similarity(sent1,sent2,stopwords=None):
    if stopwords is None:
        stopwords=[]
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1+sent2))  

    vector1 = [0] * len(all_words)  
    vector2 = [0] * len(all_words)  
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    return 1-cosine_distance(vector1,vector2)

def gen_sim_matrix(sentences,stop_words):
    similarity_matrix = np.zeros((len(sentences),len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2]=sentence_similarity(sentences[idx1],sentences[idx2],stop_words)
    
    return similarity_matrix
def generate_summary(file_name,top_n=5):
    stop_words=stopwords.words('english')
    summarize_text=[]
    sentences = read_article(file_name)
    sentence_similarity_matrix=gen_sim_matrix(sentences,stop_words)
    sentence_similarity_graph=nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence=sorted(((scores[i],s)for i,s in enumerate(sentences)),reverse=True)
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    print("Summary \n",". ".join(summarize_text))


generate_summary("sukhi.txt",1)
 '''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
