# This is a sample Python script.
import nltk
import json
import artm
from glob import glob
import preprocessing_tools as pt
import sys
import os
from os.path import join
import shutil

def preprocess():
    print("Clusterization started successfully")

    if (os.path.exists('./python/Data/norm_posts/') == False):
        os.mkdir(os.path.dirname('./python/Data/norm_posts/'))
    else:
        shutil.rmtree('./python/Data/norm_posts/')
        os.mkdir(os.path.dirname('./python/Data/norm_posts/'))

    with open('./Res/result.txt', encoding='utf-8') as f:
        wall_data = json.load(f)

    posts_by_month = []

    for i in wall_data:
        city_name = 'NA'
        if i['city']['Title']:
            city_name = i['city']['Title']
        wall = i['posts']
        clean_posts_count = 0
        for post in wall:
            raw_text = post['text']
            raw_text = raw_text.replace("\n", " ")
            if raw_text != "":
                clean_posts_count = clean_posts_count + 1

        public_text = ""
        posts_raw = []
        for post in wall:
            raw_text = post['text']
            raw_text = raw_text.replace("\n", " ")
            if raw_text != "":
                public_text = public_text + raw_text + " br "
                comments_count = 'NA'
                if post['comments']:
                    comments_count = post['comments']['Count']
                likes_count = 'NA'
                if post['likes']:
                    likes_count = post['likes']['Count']
                reposts_count = 'NA'
                if post['reposts']:
                    reposts_count = post['reposts']['Count']
                views_count = 'NA'
                if post['views']:
                    views_count = post['views']['Count']
                post_info = {
                    'group_id': i['id'],
                    'members_count': i['members_count'],
                    'name': i['name'],
                    'posts_count': len(wall),
                    'clean_posts_count': clean_posts_count,
                    'total_posts_count': i['total_count'],
                    'post_text': raw_text,
                    'comments_count': comments_count,
                    'likes_count': likes_count,
                    'reposts_count': reposts_count,
                    'views_count': views_count,
                    'date': post['date'],
                    'post_month': post['date'][:7],
                    'author_id': post['from_id'],
                    'group_city': city_name
                }
                posts_raw.append(post_info)

        norm = pt.normalize(public_text)

        posts = []
        post = []
        for txt in norm:
            if txt != '\n' and txt.strip() != '':
                if txt == 'br':
                    posts.append(post)
                    post = []
                else:
                    post.append(txt)

        for j in range(0, len(posts)):
            if posts[j] != []:
                posts_raw[j]['normalized_text'] = posts[j]

                flag = 0
                for q in range(0, len(posts_by_month)):
                    if posts_by_month[q]['month'] == posts_raw[j]['post_month']:
                        posts_by_month[q]['posts'].append(posts_raw[j])
                        flag = 1
                        break
                if flag == 0:
                    posts_by_month.append({'month': posts_raw[j]['post_month'], 'posts': []})
                    posts_by_month[len(posts_by_month) - 1]['posts'].append(posts_raw[j])

    for j in posts_by_month:
        with open('./python/Data/norm_posts/' + str(j['month']) + ".txt", 'w+t', encoding='utf-8') as f:
            if f:
                f.write(json.dumps(j['posts'], ensure_ascii=False))
            else:
                file = open('./python/Data/norm_posts/' + str(j['month']) + ".txt", 'x', encoding='utf-8')
                file.write(json.dumps(j['posts'], ensure_ascii=False))
                file.close()
    return

def GoARTM(num_topics, num_tokens, find_subclusters=0):
    dictionary_path = "./python/Data/wordbag.txt"
    vowpal_wabbit_path = "./python/Data/wordbag_Batches"

    dictionary_path_sub = "./python/Data/wordbagSub.txt"
    vowpal_wabbit_path_sub = "./python/Data/wordbag_Batches_sub"

    data_path = './python/Data/norm_posts'
    template = join(data_path, '*.txt')
    filenames = glob(template)
    result_json = [{}  for _ in range(len(filenames))]
    c = 0

    for file in filenames:
        with open(dictionary_path, 'w+t', encoding='utf-8') as f:
            f.write("")
        with open(file, 'r', encoding='utf-8') as f:
            data = f.read()
        data = json.loads(data)
        for post_info in data:
            freq = nltk.FreqDist(post_info["normalized_text"])
            text = "|text"
            z = []
            z = [(key + ":" + str(val)) for key, val in freq.items()]
            if len(z) > 0 and z[0] != ":1":
                for word in range(0, len(z)):
                    text = text + " " + z[word]
                text = text + "\n"
            with open(dictionary_path, 'a+t', encoding='utf-8') as f:
                f.write(text)

        with open(dictionary_path, 'r', encoding='utf-8') as f:
            dictTemp = f.read()
        if (os.path.exists(vowpal_wabbit_path) == True):
            shutil.rmtree(vowpal_wabbit_path)

        batch_vectorizer = artm.BatchVectorizer(data_path=dictionary_path, 
                                            data_format='vowpal_wabbit',
                                            target_folder=vowpal_wabbit_path)

        dictionary = artm.Dictionary()
        dictionary.gather(data_path=vowpal_wabbit_path)

        model = artm.ARTM(num_topics=num_topics,
                          num_document_passes=5,
                          dictionary=dictionary,
                          cache_theta=True)
        model.num_tokens = num_tokens

        model.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score_1'))
        model.scores.add(artm.SparsityThetaScore(name='sparsity_theta_score_1'))
        model.scores.add(artm.TopicKernelScore(name='topic_kernel_score_1', dictionary=dictionary))
        model.scores.add(artm.TopTokensScore(name='top_tokens_score_1', dictionary=dictionary, num_tokens=num_tokens))
        model.scores.add(artm.PerplexityScore(name='perplexity_score_1', dictionary=dictionary))

        model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse_phi_regularizer'))
        model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='sparse_theta_regularizer'))
        model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_regularizer'))

        model.initialize(dictionary=dictionary)
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=50)

        top_tokens = model.score_tracker['top_tokens_score_1']

        words = [[]  for _ in range(num_topics)]
        for i in range(num_topics):
            topic_name = model.topic_names[i]
            words[i] = [{}  for _ in range(num_tokens)]
            a = 0
            for (token, weight) in zip(top_tokens.last_tokens[topic_name], top_tokens.last_weights[topic_name]):
                words[i][a] = {
                    "word": token,
                    "weight": round(weight, 3)
                }
                a = a + 1

        with open('./python/log.txt', 'a+t', encoding='utf-8') as f:
            f.write("Keywords:" + str(words) + "\n")

        theta = model.get_theta()
        theta_arr = []
        for label, content in theta.items():
            theta_arr.append({"text": label, "values": content.sort_values(ascending=False)})
        theta_arr1 = []
        for post_info in range(0, len(data)):

            theta_arr1.append({"text": data[post_info]["post_text"],
                    "norm_text": data[post_info]["normalized_text"],
                    "group_id": data[post_info]["group_id"],
                    "group_name": data[post_info]["name"],
                    "members_count": data[post_info]["members_count"],
                    "clean_posts_count": data[post_info]["clean_posts_count"],
                    "comments_count": data[post_info]["comments_count"],
                    "likes_count": data[post_info]["likes_count"],
                    "reposts_count": data[post_info]["reposts_count"],
                    "views_count": data[post_info]["views_count"],
                    "date": data[post_info]["date"],
                    "author_id": data[post_info]["author_id"],
                    "maxVal": theta_arr[post_info]['values'][0],
                    "maxkey": theta_arr[post_info]['values'].keys()[0],
                    "assurance": theta_arr[post_info]['values'][0] - theta_arr[post_info]['values'][1],
                    "post_city": data[post_info]["group_city"]
            })

        docs_count = [0] * num_topics
        likes_count = [0] * num_topics
        reposts_count = [0] * num_topics
        views_count = [0] * num_topics
        comments_count = [0] * num_topics
        cities_info = [[]  for _ in range(num_topics)]
        assurance_avg = [0.] * num_topics
        likes_by_views = [0.] * num_topics
        posts = [[]  for _ in range(num_topics)]
        for i in theta_arr1:
            topic = int(i["maxkey"][6:])
            docs_count[topic] = docs_count[topic] + 1
            likes_count[topic] = likes_count[topic] + int(i["likes_count"])
            reposts_count[topic] = reposts_count[topic] + int(i["reposts_count"])
            views_count[topic] = views_count[topic] + int(i["views_count"])
            comments_count[topic] = comments_count[topic] + int(i["comments_count"])
            likes_by_views[topic] = (likes_by_views[topic] + float(int(i["likes_count"]) / int(i["views_count"])))
            assurance_avg[topic] = assurance_avg[topic] + i["assurance"]
            posts[topic].append(i)
            flag = 0
            if (len(cities_info[topic]) == 0):
                cities_info[topic].append({"city": i["post_city"], "docs_count": 1, "likes_count": int(i["likes_count"]), "reposts_count": int(i["reposts_count"]), "views_count": int(i["views_count"]), "comments_count": int(i["comments_count"])})
                continue
            for q in cities_info[topic]:
                if q["city"] == i["post_city"]:
                    flag = 1
                    q["docs_count"] = q["docs_count"] + 1
                    q["likes_count"] = q["likes_count"] + int(i["likes_count"])
                    q["reposts_count"] = q["reposts_count"] + int(i["reposts_count"])
                    q["views_count"] = q["views_count"] + int(i["views_count"])
                    q["comments_count"] = q["comments_count"] + int(i["comments_count"])
                    break
            if flag == 0:
                cities_info[topic].append({"city": i["post_city"], "docs_count": 1, "likes_count": int(i["likes_count"]), "reposts_count": int(i["reposts_count"]), "views_count": int(i["views_count"]), "comments_count": int(i["comments_count"])})

        for i in range(len(likes_by_views)):
            if docs_count[i] > 0:
                likes_by_views[i] = likes_by_views[i] / docs_count[i]
                assurance_avg[i] = assurance_avg[i] / docs_count[i]

        res = [[]  for _ in range(num_topics)]
        for i in range(num_topics):
            res[i] = {
                "topicName": "topic" + str(i),
                "topicWords": words[i],
                "topicDocsNumber": docs_count[i],
                "topicViewsNumber": views_count[i],
                "topicLikesNumber": likes_count[i],
                "topicRepostsNumber": reposts_count[i],
                "topicCommentsNumber": comments_count[i],
                "topicLikesByViews": round(likes_by_views[i], 3),
                "topicAssuranceAvg": round(assurance_avg[i], 3),
                "topicCitiesInfo": cities_info[i]
            }

        if(find_subclusters == 1):
            for i in range(num_topics):
                with open(dictionary_path_sub, 'w+t', encoding='utf-8') as f:
                    f.write("")
                for post_info in posts[i]:
                    freq = nltk.FreqDist(post_info["norm_text"])
                    text = "|text"
                    z = []
                    z = [(key + ":" + str(val)) for key, val in freq.items()]
                    if len(z) > 0 and z[0] != ":1":
                        for word in range(0, len(z)):
                            text = text + " " + z[word]
                        text = text + "\n"
                    with open(dictionary_path_sub, 'a+t', encoding='utf-8') as f:
                        f.write(text)

                batch_vectorizer = artm.BatchVectorizer(data_path=dictionary_path_sub,
                                                        data_format='vowpal_wabbit',
                                                        target_folder=vowpal_wabbit_path_sub)
                dictionary = artm.Dictionary()
                dictionary.gather(data_path=vowpal_wabbit_path_sub)
                dictionary.filter(min_tf=300, min_df=300)
                model.initialize(dictionary=dictionary)
                model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=50)
                top_tokens_sub = model.score_tracker['top_tokens_score_1']
                words_sub = [[] for _ in range(num_topics)]
                for q in range(0, len(model.topic_names)):
                    topic_name = model.topic_names[q]
                    words_sub[q] = [{} for _ in range(num_tokens)]
                    a = 0
                    for (token, weight) in zip(top_tokens_sub.last_tokens[topic_name], top_tokens_sub.last_weights[topic_name]):
                        words_sub[q][a] = {
                            "word": token,
                            "weight": round(weight, 3)
                        }
                        a = a + 1
                theta_sub = model.get_theta()
                theta_arr_sub = []
                for label, content in theta_sub.items():
                    theta_arr_sub.append({"text": label, "values": content.sort_values(ascending=False)})
                theta_arr1_sub = []
                for post_info in range(0, len(posts[i])):
                    theta_arr1_sub.append({"text": posts[i][post_info]["text"],
                        "norm_text": posts[i][post_info]["norm_text"],
                        "group_id": posts[i][post_info]["group_id"],
                        "group_name": posts[i][post_info]["group_name"],
                        "members_count": posts[i][post_info]["members_count"],
                        "clean_posts_count": posts[i][post_info]["clean_posts_count"],
                        "comments_count": posts[i][post_info]["comments_count"],
                        "likes_count": posts[i][post_info]["likes_count"],
                        "reposts_count": posts[i][post_info]["reposts_count"],
                        "views_count": posts[i][post_info]["views_count"],
                        "date": posts[i][post_info]["date"],
                        "author_id": posts[i][post_info]["author_id"],
                        "maxVal": theta_arr_sub[post_info]['values'][0],
                        "maxkey": theta_arr_sub[post_info]['values'].keys()[0],
                        "assurance": theta_arr_sub[post_info]['values'][0] - theta_arr_sub[post_info]['values'][1]
                    })
                docs_count_sub = [0] * num_topics
                likes_count_sub = [0] * num_topics
                reposts_count_sub = [0] * num_topics
                views_count_sub = [0] * num_topics
                comments_count_sub = [0] * num_topics
                groups_info_sub = [[] for _ in range(num_topics)]
                assurance_avg_sub = [0.] * num_topics
                likes_by_views_sub = [0.] * num_topics
                for g in theta_arr1_sub:
                    topic = int(g["maxkey"][6:])
                    docs_count_sub[topic] = docs_count[topic] + 1
                    likes_count_sub[topic] = likes_count[topic] + int(g["likes_count"])
                    reposts_count_sub[topic] = reposts_count[topic] + int(g["reposts_count"])
                    views_count_sub[topic] = views_count[topic] + int(g["views_count"])
                    comments_count_sub[topic] = comments_count[topic] + int(g["comments_count"])
                    likes_by_views_sub[topic] = (likes_by_views_sub[topic] + float(int(g["likes_count"]) / int(g["views_count"])))
                    assurance_avg_sub[topic] = assurance_avg_sub[topic] + g["assurance"]
                    flag = 0
                    if (len(groups_info_sub[topic]) == 0):
                        groups_info_sub[topic].append(
                            {"group": g["group_name"], "docs_count": 1, "likes_count": int(g["likes_count"]),
                             "reposts_count": int(g["reposts_count"]), "views_count": int(g["views_count"]),
                             "comments_count": int(g["comments_count"])})
                        continue
                    for q in groups_info_sub[topic]:
                        if q["group"] == g["group_name"]:
                            flag = 1
                            q["docs_count"] = q["docs_count"] + 1
                            q["likes_count"] = q["likes_count"] + int(g["likes_count"])
                            q["reposts_count"] = q["reposts_count"] + int(g["reposts_count"])
                            q["views_count"] = q["views_count"] + int(g["views_count"])
                            q["comments_count"] = q["comments_count"] + int(g["comments_count"])
                            break
                    if flag == 0:
                        groups_info_sub[topic].append(
                            {"group": g["group_name"], "docs_count": 1, "likes_count": int(g["likes_count"]),
                             "reposts_count": int(g["reposts_count"]), "views_count": int(g["views_count"]),
                             "comments_count": int(g["comments_count"])})
                for g in range(len(likes_by_views_sub)):
                    if docs_count_sub[g] > 0:
                        likes_by_views_sub[g] = likes_by_views_sub[g] / docs_count_sub[g]
                        assurance_avg_sub[g] = assurance_avg_sub[g] / docs_count_sub[g]
                res_sub = [[] for _ in range(num_topics)]
                for g in range(num_topics):
                    res_sub[g] = {
                        "topicName": "topic" + str(g),
                        "topicWords": words_sub[g],
                        "topicDocsNumber": docs_count_sub[g],
                        "topicViewsNumber": views_count_sub[g],
                        "topicLikesNumber": likes_count_sub[g],
                        "topicRepostsNumber": reposts_count_sub[g],
                        "topicCommentsNumber": comments_count_sub[g],
                        "topicLikesByViews": round(likes_by_views_sub[g], 3),
                        "topicAssuranceAvg": round(assurance_avg_sub[g], 3),
                        "topicGroupsInfo": groups_info_sub[g]
                    }
                res[i]['topicSubClusters'] = res_sub
        result_json[c] = {"month": str(file[-11:-4]), "data": res}
        c = c + 1

    if (os.path.exists('./python/Data/Result/') == False):
        os.mkdir(os.path.dirname('./python/Data/Result/'))

    with open('./python/Data/Result/ClusterizationResults.txt', 'w+t', encoding='utf-8') as f:
        f.write(json.dumps(str(result_json), ensure_ascii=False))
    print("Clusterization completed successfully")
    return

def GoLDA(num_topics, num_tokens):
    dictionary_path = "./python/Data/wordbag.txt"
    vowpal_wabbit_path = "./python/Data/wordbag_Batches"

    data_path = './python/Data/norm_posts'
    template = join(data_path, '*.txt')
    filenames = glob(template)
    result_json = [{} for _ in range(len(filenames))]
    c = 0
    for file in filenames:
        month = file[-11:-4]
        print(month)
        with open(dictionary_path, 'w+t', encoding='utf-8') as f:
            f.write("")
        with open(file, 'r', encoding='utf-8') as f:
            data = f.read()
        data = json.loads(data)
        for post_info in data:
            freq = nltk.FreqDist(post_info["normalized_text"])
            text = "|text"
            z = []
            z = [(key + ":" + str(val)) for key, val in freq.items()]
            if len(z) > 0 and z[0] != ":1":
                for word in range(0, len(z)):
                    text = text + " " + z[word]
                text = text + "\n"
            with open(dictionary_path, 'a+t', encoding='utf-8') as f:
                f.write(text)

        batch_vectorizer = artm.BatchVectorizer(data_path=dictionary_path,
                                            data_format='vowpal_wabbit',
                                            target_folder=vowpal_wabbit_path)

        dictionary = artm.Dictionary()
        dictionary.gather(data_path=vowpal_wabbit_path)

        model = artm.LDA(num_topics=num_topics,
                          num_document_passes=5,
                          dictionary=dictionary,
                          cache_theta=True)

        model.num_tokens = num_tokens
        model.initialize(dictionary=dictionary)
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=50)

        topics = model.get_top_tokens(num_tokens=num_tokens, with_weights=True)
        theta = model.get_theta()  # .transpose()
        docs_num = len(model.get_theta().transpose())
        theta_arr = []
        for label, content in theta.items():
            theta_arr.append({"text": label, "values": content.sort_values(ascending=False)})
        theta_arr1 = []
        for post_info in range(0, len(data)):
            theta_arr1.append({"maxkey": theta_arr[post_info]['values'].keys()[0],
                               "assurance": theta_arr[post_info]['values'][0] - theta_arr[post_info]['values'][1]
            })

        docs_count = [0] * num_topics
        assurance_avg = [0.] * num_topics
        for i in theta_arr1:
            topic = int(i["maxkey"][6:])
            docs_count[topic] = docs_count[topic] + 1
            assurance_avg[topic] = assurance_avg[topic] + i["assurance"]

        for i in range(len(docs_count)):
            if docs_count[i] > 0:
                assurance_avg[i] = assurance_avg[i] / docs_count[i]

        res = [[]] * num_topics
        for i in range(num_topics):
            res[i] = {
                "topicName": "topic" + str(i),
                "topicWords": topics[i],
                "topicDocsNumber": docs_count[i],
                "topicAssuranceAvg": assurance_avg[i]
            }

        result_json[c] = {"month": str(file[-11:-4]), "data": res}
        c = c + 1

    with open('./python/Data/Result/LDAResults.txt', 'w+t', encoding='utf-8') as f:
        f.write(json.dumps(str(result_json), ensure_ascii=False))
    return

def findPostsBySubstr(substr):
    substr = substr.lower()
    print("Search started successfully")
    with open('./Res/result.txt', encoding='utf-8') as f:
        wall_data = json.load(f)
    posts_by_month = []
    for i in wall_data:
        city_name = 'NA'
        if i['city']['Title']:
            city_name = i['city']['Title']
        wall = i['posts']
        clean_posts_count = 0
        for post in wall:
            raw_text = post['text']
            raw_text = raw_text.replace("\n", " ")
            if raw_text != "":
                clean_posts_count = clean_posts_count + 1

        public_text = ""
        posts_raw = []
        for post in wall:
            raw_text = post['text']
            raw_text = raw_text.replace("\n", " ")
            if raw_text != "":
                public_text = public_text + raw_text + " br "
                comments_count = 'NA'
                if post['comments']:
                    comments_count = post['comments']['Count']
                likes_count = 'NA'
                if post['likes']:
                    likes_count = post['likes']['Count']
                reposts_count = 'NA'
                if post['reposts']:
                    reposts_count = post['reposts']['Count']
                views_count = 'NA'
                if post['views']:
                    views_count = post['views']['Count']
                post_info = {
                    'group_id': i['id'],
                    'members_count': i['members_count'],
                    'name': i['name'],
                    'posts_count': len(wall),
                    'clean_posts_count': clean_posts_count,
                    'total_posts_count': i['total_count'],
                    'post_text': raw_text,
                    'comments_count': comments_count,
                    'likes_count': likes_count,
                    'reposts_count': reposts_count,
                    'views_count': views_count,
                    'date': post['date'],
                    'post_month': post['date'][:7],
                    'author_id': post['from_id'],
                    'group_city': city_name
                }
                posts_raw.append(post_info)

        public_text = public_text + substr + " br "
        norm = pt.normalize(public_text)
        substr = norm[-2]
        norm = norm[:-2]
        posts = []
        post = []
        j = 0
        for txt in norm:
            if txt != '\n' and txt.strip() != '':
                if txt == 'br':
                    for word in post:
                        if (word == substr):
                            posts_raw[j]['normalized_text'] = post
                            posts.append(posts_raw[j])
                            break
                    post = []
                    j = j + 1
                else:
                    post.append(txt)
        for j in range(0, len(posts)):
            flag = 0
            for q in range(0, len(posts_by_month)):
                if posts_by_month[q]['month'] == posts[j]['post_month']:
                    flag = 1
                    flag1 = 0
                    for y in range(0, len(posts_by_month[q]['cities'])):
                        if posts_by_month[q]['cities'][y]["city"] == posts[j]["group_city"]:
                            flag1 = 1
                            posts_by_month[q]['cities'][y]["docs_count"] = posts_by_month[q]['cities'][y]["docs_count"] + 1
                            posts_by_month[q]['cities'][y]["likes_count"] = posts_by_month[q]['cities'][y]["likes_count"] + int(posts[j]["likes_count"])
                            posts_by_month[q]['cities'][y]["reposts_count"] = posts_by_month[q]['cities'][y]["reposts_count"] + int(posts[j]["reposts_count"])
                            posts_by_month[q]['cities'][y]["views_count"] = posts_by_month[q]['cities'][y]["views_count"] + int(posts[j]["views_count"])
                            posts_by_month[q]['cities'][y]["comments_count"] = posts_by_month[q]['cities'][y]["comments_count"] + int(posts[j]["comments_count"])
                            break
                    if flag1 == 0:
                        posts_by_month[q]['cities'].append({"city": posts[j]["group_city"], "docs_count": 1, "likes_count": int(posts[j]["likes_count"]), "reposts_count": int(posts[j]["reposts_count"]), "views_count": int(posts[j]["views_count"]), "comments_count": int(posts[j]["comments_count"])})
                        break
            if flag == 0:
                posts_by_month.append({'month': posts[j]['post_month'], 'cities': []})
                posts_by_month[len(posts_by_month) - 1]['cities'].append({"city": posts[j]["group_city"], "docs_count": 1, "likes_count": int(posts[j]["likes_count"]), "reposts_count": int(posts[j]["reposts_count"]), "views_count": int(posts[j]["views_count"]), "comments_count": int(posts[j]["comments_count"])})

    result = {"search": substr, "result": posts_by_month}

    if (os.path.exists('./python/Data/Result/') == False):
        os.mkdir(os.path.dirname('./python/Data/Result/'))

    with open('./python/Data/Result/SearchResults.txt', 'w+t', encoding='utf-8') as f:
        f.write(json.dumps(str(result), ensure_ascii=False))

    print("Search completed successfully")

    return

def main():
    lc = artm.messages.ConfigureLoggingArgs()
    lc.minloglevel = 2
    lib = artm.wrapper.LibArtm(logging_config=lc)

    num_topics = int(sys.argv[1])
    num_tokens = int(sys.argv[2])
    #substr = str(sys.argv[1])

    #findPostsBySubstr(substr)
    preprocess()
    GoARTM(num_topics, num_tokens)
    #GoLDA(num_topics, num_tokens)

if __name__ == '__main__':
    main()