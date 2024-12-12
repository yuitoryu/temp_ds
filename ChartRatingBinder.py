# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:34:04 2024

@author: Yingqi Bian
"""


from pathlib import Path
import json
from mutagen.mp3 import MP3

def matching_diff(text, rating, dataset, correspond, song_length):
    # text: txt content of a song, including chart of different difficulty
    # rating: a dict that stores ratings of chart
    name = text[0][7:-4]
    category = text[0][-3:-1]
    song = {'name' : name + category,
            'song_length' : song_length,
            'difficulty': {}}
    #print(name)
    song_exist = rating.get(name, 0)
    #print(song_exist)
    if song_exist != 0:
        #print('yes')
        song_exist = song_exist[category]    
        for keyword in correspond.keys():
            #print(keyword)
            if keyword in text:
                #print('yes')
                chart = extract_chart(text, keyword)
                rating_num = song_exist[correspond[keyword]]
                song['difficulty'][correspond[keyword]] = {}
                song['difficulty'][correspond[keyword]]['rating_num'] = rating_num
                #print(song)
                song['difficulty'][correspond[keyword]]['chart'] = chart
                
        dataset.append(song)



def extract_chart(text, keyword):
    # text: txt content of a song, including chart of different difficulty
    # keyword: in form of &inote_
    cur = text.index(keyword) + 1
    chart = []
    while  text[cur] != 'E':
        chart.append(text[cur])
        cur += 1
    return chart


def extract(folder_path, rating, save_dir=None):
    #folder_path = Path(r"D:\maimai")
    folder_path = Path(folder_path)
    directories = [str(d) for d in folder_path.iterdir() if d.is_dir()]
    #json_file = open('D:/maimai_teisu_analyzer_project/rating.json','r',encoding='utf-8')
    if isinstance(rating, str):
        json_file = open(rating,'r',encoding='utf-8')
        rating = json.load(json_file)
    dataset=[]
    correspond = {'&inote_2=':'Basic', '&inote_3=':'Advanced', '&inote_4=':'Expert',
                  '&inote_5=':'Master', '&inote_6=':'Re:MASTER'}
    
    
    for directory in directories:
        #print(directory)
        fp = open(directory+'/maidata.txt' ,'r', encoding='utf-8')
        text = fp.readlines()
        for i in range(len(text)):
            text[i] = text[i].strip('\n')
            
        #get song length
        # Load the mp3 file
        audio = MP3(directory+'/track.mp3')
        # Get the length of the audio in seconds
        song_length = audio.info.length
        
        matching_diff(text, rating, dataset, correspond, song_length)
        #print(text)
    if save_dir != None:
        with open(save_dir,'w',encoding='utf-8') as result:
            json.dump(dataset, result, indent=4)
        
    return dataset

def bind_rate(csv_file):
    fp =  open(csv_file, 'r', encoding='utf-8')
    data = fp.readlines()
    del data[0]
    level_dict = {}
    #写入歌曲定数
    for song in data:
        song.strip('\n')
        temp = song.split(',')
    
        # create if song does not exist
        # song name->std/dx->basic/advance/expert/master->rating
        if len(temp) >= 8:
            name = ''.join(temp[:-7])
            temp = [name] + temp[-7:]
        if level_dict.get(temp[0],0) == 0:
            level_dict[temp[0]] = {}
            level_dict[temp[0]]['SD'] = {}
            level_dict[temp[0]]['DX'] = {}
    
            for diff in ['Basic','Advanced','Expert','Master','Re:MASTER']:
                level_dict[temp[0]]['SD'][diff] = None
                level_dict[temp[0]]['DX'][diff] = None
    
        # record difficulty of current chart
        level_dict[temp[0]][temp[1]][temp[2]] = float(temp[4])
    return level_dict