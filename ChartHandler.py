# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:05:53 2024

@author: Yingqi Bian
"""
import progressbar
import time
import numpy as np
import re
import json
class chartDecomposer():
    # Decompose a chart (string with several lines) into pieces and interpret. Data stored in a nested list
    def __init__(self): #list where each element stands for a line
        self.bpm = []
        self.name = None
        # rating
        self.rating_num = 0
        # V slide reference matrix
        self.begin_to_mid = np.array([[0,0,1,0,0,0,1,0],
                                      [0,0,0,1,0,0,0,1],
                                      [1,0,0,0,1,0,0,0],
                                      [0,1,0,0,0,1,0,0],
                                      [0,0,1,0,0,0,1,0],
                                      [0,0,0,1,0,0,0,1],
                                      [1,0,0,0,1,0,0,0],
                                      [0,1,0,0,0,1,0,0]])
        self.mid_to_end = np.array([[0,0,7,9,9,9,3,0],
                                    [0,0,0,8,9,9,9,4],
                                    [5,0,0,0,1,9,9,9],
                                    [9,6,0,0,0,2,9,9],
                                    [9,9,7,0,0,0,3,9],
                                    [9,9,9,8,0,0,0,4],
                                    [5,9,9,9,1,0,0,0],
                                    [0,6,9,9,9,2,0,0]])
        
        # Tap recorder
        self.tap = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        
        # Hold recorder
        self.hold = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}

        # Touch (hold) recorder
        self.A = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        self.B = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        self.C = {1:[],2:[]}
        self.Ch = []
        self.D = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        self.E = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        self.touch = {'A':self.A, 'B':self.B, 'C':self.C, 'D':self.D, 'E':self.E} 
        
        # Slide recorder
        self.slide = {}
        for begin in range(1,9):
            self.slide[begin] = {}
            # Slide type            
            for shape in ['-', '^', '<', '>', 'v', 'p', 'q', 's', 'z', 'w', 'pp', 'qq']:
                self.slide[begin][shape] = {}
                
            for end in range(1,9):
                # straight '-'
                if abs(begin-end) != 0 and abs(begin-end) != 1 and abs(begin-end) != 7:
                    self.slide[begin]['-'][end] = []
                
                #Short arc '^'、'v'
                if begin != end and abs(begin-end) != 4:
                    self.slide[begin]['^'][end] = []
                    self.slide[begin]['v'][end] = []

                # '<'，'>'，‘p’，‘q’，‘pp’，‘qq’
                for shape in ['<','>','p','q','pp','qq']:
                    self.slide[begin][shape][end] = []

                # 's','z',wifi'w'
                for shape in ['s','z','w']:
                    if abs(begin-end) == 4:
                        self.slide[begin][shape][end] = []


            # Grand 'V'
            self.slide[begin]['V'] = {}
            for mid in range(1,9):
                if self.begin_to_mid[begin-1][mid-1] == 1:
                    self.slide[begin]['V'][mid] = {}
                    for end in range(1,9):
                         if self.mid_to_end[mid-1][end-1] == 9 or self.mid_to_end[mid-1][end-1] == begin:
                             self.slide[begin]['V'][mid][end] = []

    def remove_consecutive_duplicates(self, input_list):
        # For somehow, there is just duplicates element after decomposition
        # This is to fix it
        if not input_list: # empty case
            return []

        result = [input_list[0]]  # Initialize with the first element
        for i in range(1, len(input_list)):
            if input_list[i] != input_list[i - 1]:  # Compare with the previous element
                result.append(input_list[i])

        return result

    def remove_repeated_slide(self):
        # Parse through each list nested in the nested dictionary to remove duplicate component
        for begin in self.slide.keys():
                for shape in self.slide[begin].keys():
                    if shape != 'V':
                        for end in self.slide[begin][shape].keys():
                            self.slide[begin][shape][end] = self.remove_consecutive_duplicates(self.slide[begin][shape][end])
                    else:
                        for mid in self.slide[begin][shape].keys():
                            for end in self.slide[begin][shape][mid].keys():
                                self.slide[begin][shape][mid][end] = self.remove_consecutive_duplicates(self.slide[begin][shape][mid][end])
    
    def output_data(self):
        # output all tokens and rating attached to that
        self.remove_repeated_slide()
        note = {'tap' : self.tap,
                'hold' : self.hold,
                'touch' : self.touch,
                'Ch' : self.Ch,
                'slide' : self.slide}
        return {'name' : self.name, 'rating_num' : self.rating_num, 'bpm' : self.bpm, 'note' : note}
        
    def fetch_slice_in_string(self, text, start_index, target):
        # Get position of specific text from a string
        if start_index == len(text)-1:
            return [text[start_index], start_index]
        else:
            if text[start_index] == target:
                return [target, start_index]
            else:
                result = self.fetch_slice_in_string(text, start_index + 1, target)
                return [text[start_index] + result[0], result[1]]

    def take_out_piece_by_index(self, text, start_index, end_index):
        # Extract out text by taking out text with targetd position
        first_half = text[:start_index]
        if end_index == len(text)-1:
            second_half = ''
        else:
            second_half = text[end_index+1:]
        return first_half + second_half

    
    def decompose(self, fullchart, name): # database[n][difficulty][diff]
        # Decomposing the chart
        self.name = name
        cur_time = 0
        self.rating_num = fullchart['rating_num']
        for line in fullchart['chart']:
            cur_time = self.decompose_line(line, cur_time)

    def decompose_line(self, line, cur_time):
        # Decompose a single line of the chart
        element = line
        #Find beat type
        assert element.find('{') != -1, '几分音信息缺失'
        [beat_frac_text, end_index] = self.fetch_slice_in_string(element, element.find('{'), '}')
        beat_frac = int(beat_frac_text.strip('{').strip('}'))
        element = self.take_out_piece_by_index(element, element.find('{'), end_index)
        elements = element.split(',')
        del elements[-1]
        cur_bpm = 0
        for element in elements:
            cur_time = self.decompose_same_time_element(element, beat_frac, cur_time, cur_bpm)
        return cur_time

    
    def decompose_same_time_element(self, element, beat_frac, cur_time, cur_bpm):
        # Handle a single note
        notes = element.split('/')
        for note in notes:
            self.analyze_single_element(note, beat_frac, cur_time, cur_bpm)
        cur_time += 1/beat_frac
        return cur_time

    def analyze_single_element(self, note, beat_frac, cur_time, cur_bpm):
        # Read bpm
        if len(note) > 0:
            if note[0] == '(':
                bpm_text = note[1:note.find(')')]
                note = note[note.find(')')+1:]
                cur_bpm = float(bpm_text)
                self.bpm.append((cur_time, cur_bpm))
        
        # Read tap
        if len(note) == 1 and note[0].isnumeric():
            self.tap[int(note[0])].append((cur_time, 0, 0))
            return cur_bpm

        # Read ex tap
        if len(note) == 2:
            if note[0].isnumeric() and note[1] == 'x':
                self.tap[int(note[0])].append((cur_time, 0, 1))
                return cur_bpm

        # Read break tap
        if len(note) == 2: 
            if note[0].isnumeric() and note[1] == 'b':
                self.tap[int(note[0])].append((cur_time, 1, 0))
                return cur_bpm

        # Read break ex tap
        if len(note) == 3:
            if note[0].isnumeric() and note[1] == 'b' and note[2] == 'x':
                self.tap[int(note[0])].append((cur_time, 1, 1))
                return cur_bpm

        # Read hold
        if len(note) > 3:
            if note[0].isnumeric() and note[1:3] == 'h[':
                beat_frac, num_of_frac = self.get_duration(note[2:])
                self.hold[int(note[0])].append((cur_time, num_of_frac / beat_frac, 0, 0))
                return cur_bpm

        # Read ex hold
        if len(note) > 4:
            if note[0].isnumeric() and 'h' in note[1:3] and 'x' in note[1:3] and note[3] == '[':
                beat_frac, num_of_frac = self.get_duration(note[3:])
                self.hold[int(note[0])].append((cur_time, num_of_frac / beat_frac, 0, 1))
                return cur_bpm

        # Read break hold
        if len(note) > 4:
            if note[0].isnumeric() and 'h' in note[1:3] and 'b' in note[1:3] and note[3] == '[':
                beat_frac, num_of_frac = self.get_duration(note[3:])
                self.hold[int(note[0])].append((cur_time, num_of_frac / beat_frac, 1, 0))
                return cur_bpm

        # Read break ex hold
        if len(note) > 5:
            if note[0].isnumeric() and 'h' in note[1:4] and 'b' in note[1:4] and 'x' in note[1:4] and note[4] == '[':
                beat_frac, num_of_frac = self.get_duration(note[4:])
                self.hold[int(note[0])].append((cur_time, num_of_frac / beat_frac, 1, 1))
                return cur_bpm

        # Read touch
        if len(note) == 2 or len(note) == 3:
            if note[0].isalpha() and note[1].isnumeric():
                self.touch[note[0]][int(note[1])].append((cur_time, 1))
                return cur_bpm

        # Read touch hold
        if len(note) >= 3:
            if (note[0:3] == 'Ch[' or note[0:4] == 'C1h[' or note[0:4] == 'C2h['  or note[0:5] == 'C1fh[' 
                or note[0:5] == 'C2fh[' or note[0:5] == 'C1hf[' or note[0:5] == 'C2hf['):
                left_bracket = re.compile(r'\[')
                right_bracket = re.compile(r'\]')
                begin = 0
                end = 0
                for idx in left_bracket.finditer(note):
                    begin = idx.start()
                for idx in right_bracket.finditer(note):
                    end = idx.end()
                beat_frac, num_of_frac = self.get_duration(note[begin:end])
                self.Ch.append((cur_time, num_of_frac / beat_frac))
                return cur_bpm

        # Read slide
        if len(note) >= 5:
            slides = note.split('*')
            head, slide = self.chop_head(slides[0])
            do_register_head = True
            for slide in slides:
                if not slide[0].isnumeric():
                    slide = head + slide

                head, slide = self.chop_head(slide)
                self.decompose_slides(head, slide, cur_time, cur_bpm, do_register_head)
                do_register_head = False
                return cur_bpm
                
        return cur_bpm
 
                
    def chop_head(self, slide):
        flag = True
        cur = 0
        while flag:
            cur += 1
            flag = slide[cur] == 'b' or slide[cur] == 'x'
        return slide[:cur], slide[cur:]

    def decompose_slides(self, head, slide, cur_time, cur_bpm, do_register_head):
        # Decompose slide
        # '-', '^', '<', '>', 'v', 'p', 'q', 's', 'z', 'w', 'pp', 'qq'
        node = re.compile(r'(qq|pp|-|\^|<|>|v|p|q|s|z|w|V)[1-8]([1-8]|)(b|)(\[|)') #shape + position + [
        matches = node.finditer(slide)
        
        node_store = []
        for match in matches:
            target = slide[ match.start() : match.end()]
            if target[-1] == '[':
                target = target[:-1]
            node_store.append(target)
        
        left_bracket = re.compile(r'\[')
        right_bracket = re.compile(r'\]')
        left_store = []
        right_store = []
        interval_store = []
        left_match = left_bracket.finditer(slide)
        right_match = right_bracket.finditer(slide)
        
        for left in left_match:
            left_store.append(left.start())
        for right in right_match:
            right_store.append(right.end())
        for i in range(len(left_store)):
            interval_store.append(slide[ left_store[i] : right_store[i] ])
        timing = []
        '''
        Waiting time is one beat at 160 BPM, tracing length is three 8th notes at 160 BPM ... 【1-4[160#8:3],】
        Waiting time is one beat at 160 BPM, tracing length is 2 seconds ... 【1-4[160#2],】
        Waiting time is 3 seconds, tracing length is 1.5 seconds ... 【1-4[3##1.5],】
        Waiting time is 3 seconds, tracing length is three 8th notes at current BPM ... 【1-4[3##8:3],】
        Waiting time is 3 seconds, tracing length is three 8th notes at 160 BPM ... 【1-4[3##160#8:3],】
        '''
        for interval in interval_store:
            delay = 0
            if '#' in interval:
                if ':' not in interval: # slide in second
                    if '##' in interval: # delay in second
                        [delay, duration] = interval.strip('[').strip(']').split('##')
                        delay = float(delay) / 60 * cur_bpm / 4
                    else: # dealy in beat
                        [delay, duration] = interval.strip('[').strip(']').split('#')
                        delay = cur_bpm / float(delay) / 4
                    duration = float(duration) / 60 * cur_bpm / 4
                    timing.append(duration)
                else:
                    if '##' not in interval:
                        [delay, duration] = interval.strip('[').strip(']').split('##')
                        delay = cur_bpm / float(delay) / 4
                        beat_frac, num_of_frac = self.get_duration(duration)
                        timing.append(num_of_frac / beat_frac)
                    else:
                        pattern = re.compile(r'([0-9]|)[0-9]##([0-9]|)([0-9]|)([0-9]|)([0-9]):([0-9]|)([0-9]|)([0-9]|)([0-9])')
                        result = pattern.search(interval)
                        if result:
                            [delay, duration] = result.group().split('##')
                            delay = float(delay) / 60 * cur_bpm / 4
                            duration = '[' + duration + ']'
                            beat_frac, num_of_frac = self.get_duration(duration)
                            timing.append(num_of_frac / beat_frac)
                        else:
                            [delay, second_half] = interval.strip('[').strip(']').split('##')
                            [bpm, duration] = second_half.split('#')
                            delay = float(delay) / 60 * cur_bpm / 4
                            duration = '[' + duration + ']'
                            beat_frac, num_of_frac = self.get_duration(duration)
                            timing.append(num_of_frac / beat_frac*cur_bpm/float(bpm))
            else:
                delay = 1
                beat_frac, num_of_frac = self.get_duration(interval)
                timing.append(num_of_frac / beat_frac)
            if len(timing) == 1:
                duration = timing[0] / len(node_store)
                timing = [duration for i in range(len(node_store))]

            # create slide object and make info_block
            this_chained_slide = chainedSlideRegister()
            this_chained_slide.compose_chain(head, node_store, timing)
            info_dicts = this_chained_slide.prepare_info_blocks()

            # adding slide tokens to the dictionary
            temp_time = cur_time
            for i in range(len(info_dicts)):
                #{'begin':self.begin, 'shape':self.shape, 'end':self.end, 'duration':self.duration}
                begin = info_dicts[i]['begin']
                shape = info_dicts[i]['shape']
                mid = info_dicts[i].get('mid',0)
                end = info_dicts[i]['end']
                duration = info_dicts[i]['duration']
                # Grand V
                if mid != 0:
                    self.slide[begin][shape][mid][end].append((temp_time + delay, duration, int(this_chained_slide.is_break())))
                # Other slide
                else:
                    self.slide[begin][shape][end].append((temp_time + delay, duration, int(this_chained_slide.is_break())))
                temp_time += duration

        # Add tap for slide
        if do_register_head:
            begin = this_chained_slide.head
            self.tap[begin].append((cur_time, int(this_chained_slide.bhead), int(this_chained_slide.xhead)))

    def get_duration(self, text):
        #text: [num:num]
        beat_frac = text[ 1 : text.find(':') ]
        num_of_frac = text[ text.find(':')+1 : -1]
        beat_frac = int(beat_frac)
        num_of_frac = int(num_of_frac)        
        return beat_frac, num_of_frac
        
    
class basicSlideRegister():
    # The class for handling slide note (slide part, not including the tap at beginning)
    def __init__(self, shape):
        self.begin = None
        self.shape = shape
        self.end = None
        self.duration = None
        self.delay = 0

    def register(self, begin, end, duration):
        self.begin = int(begin)
        self.end = int(end)
        self.duration = duration

    def prepare_info_block(self):
        return {'begin':self.begin, 'shape':self.shape, 'end':self.end, 'duration':self.duration}



class grandVSlideRegister(basicSlideRegister):
    # Handle the grand V shape slide that has a mid parameter
    def __init__(self):
        super().__init__('V')
        self.mid = None

    def register(self, begin, mid, end, duration):
        self.begin = int(begin)
        self.mid = int(mid)
        self.end = int(end)
        self.duration = duration

    def prepare_info_block(self):
        return {'begin':self.begin, 'shape':self.shape, 'mid':self.mid, 'end':self.end, 'duration':self.duration}
    
class chainedSlideRegister():
    # Handle chained type slide
    # Decompose everything to several simple basic slides
    def __init__(self):
        self.chain = []
        self.juezan = False
        self.head = None
        self.bhead = False
        self.xhead = False

    def is_break(self):
        return self.juezan

    def is_break_head(self):
        return self.bhead

    def is_ex_head(self):
        return self.xhead
        
    def compose_chain(self, head, node_store, interval_store):
        self.head = int(head[0])
        # initialize lists for storing begin, shape and end data for chains of slides
        begin = [head[0]]
        shapes = []
        end = []
        shape = re.compile(r'(qq|pp|-|\^|<|>|v|p|q|s|z|w|V)')
        for node in node_store:
            matches = shape.finditer(node)
            for match in matches:
                shapes.append( node[ match.start() : match.end() ] )
                position = node[match.end():]
                # is break slide or not
                if 'b' in position:
                    self.juezan = True
                    position = position[:-1] # remove break notation
                begin.append(position)
                end.append(position)
        del begin[-1]
        
        for i in range(len(shapes)):
            if shapes[i] != 'V':
                cur_slide = basicSlideRegister(shapes[i])
                cur_slide.register(begin[i][-1], end[i][-1], interval_store[i])
            else:
                cur_slide = grandVSlideRegister()
                cur_slide.register(begin[i][-1], end[i][0], end[i][-1], interval_store[i])
            self.chain.append(cur_slide)

        if 'b' in head:
            self.bhead = True
        if 'x' in head:
            self.xhead = True
        self.head = int(head[0])
            
    def prepare_info_blocks(self):
        return [slide.prepare_info_block() for slide in self.chain]
           
def decompose_charts(database, save_dir, progress_bar_on=False):
    if isinstance(database, str):
        with open(database, 'r', encoding='utf-8') as fp:
            database = json.load(fp)
    elif not isinstance(database, list):
        raise TypeError('The input database is neither a list nor a file path')
        
    if progress_bar_on:
        counter = 0
        for song in database:
            for diff in song['difficulty']:
                counter += 1
        bar = progressbar.ProgressBar(maxval=counter, widgets=[
                                  progressbar.Bar('=', '[', ']'), 
                                  ' ', 
                                  progressbar.Percentage(), 
                                  ' (', progressbar.SimpleProgress(), ')'
                              ])
    processed_data = []    
    for song in database:
    #print(song['name'])
        for diff in song['difficulty']:
            #print(diff)
            fullchart = song['difficulty'][diff]
            current = chartDecomposer()
            current.decompose(fullchart, song['name'])
            this_song_info = current.output_data()
            processed_data.append(this_song_info)
            if progress_bar_on:
                bar.update()
                
    with open(save_dir, 'w') as final_store:
        json.dump(processed_data, final_store, indent=4)
    
    return processed_data
            
class slideToSensorConverter():
    # Decompose a slide to sensor areas on the screen
    # Significantly reduced dim of tokens at the end
    def __init__(self):
        # Create a mapping for methods that handle different shape of slides
        self.func = {'-': self.straight,
                     '>': self.clockwise,
                     '<': self.left_arc_slide,
                     '>': self.right_arc_slide,
                     'p': self.pslide,
                     'q': self.qslide,
                     'pp': self.ppslide,
                     'qq': self.qqslide,
                     'w': self.wifi,
                     's': self.sslide,
                     'z': self.zslide,
                     'v': self.vslide,
                     'V': self.grandvslide}

    def convert(self, slide_dict):
        # The conversion method
        res = {'A':{1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]},
               'B':{1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}, 
               'C':{1:[]},
               'D':{1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}, 
               'E':{1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}}
        for begin in slide_dict.keys():
            # Pick method based on shape
            for type in slide_dict[begin].keys():
                if type != 'V':
                    for end in slide_dict[begin][type].keys():
                        for slide in slide_dict[begin][type][end]:
                            touch = self.func[type](int(begin), int(end))
                            start = slide[0]
                            duration = slide[1]
                            self.assign(res, touch, type, start, duration, self.is_break(slide))
                else:
                    for mid in slide_dict[begin][type].keys():
                        for end in slide_dict[begin][type][mid].keys():
                            for slide in slide_dict[begin][type][mid][end]:
                                touches = self.func[type](int(begin), int(mid), int(end))
                                start = slide[0]
                                duration = slide[1]
                                self.assign(res, touches, type, start, duration, self.is_break(slide))
        #print('return')
        return res

    def is_break(self, slide):
        return slide[-1]
    
    def assign(self, touch_dict, touches, type, start, duration, is_break):
        # The wifi shape slide has to be handled separated since it involves activating sensor ares at the same time
        # For others, sensor areas are activated one by another
        if type == 'w':  
            self.assign_wifi(touch_dict, touches, start, duration, is_break)
            return ;
        step = duration / len(touches)
        cur = start
        for touch in touches:
            if isinstance(touch[0], list):
                for item in touch:
                    touch_dict[item[0]][item[1]].append([cur, 1/len(touch), is_break])
            else:
                touch_dict[touch[0]][touch[1]].append([cur, 1, is_break])
            cur += step

    def assign_wifi(self, touch_dict, touches, start, duration, is_break):
        # Handle wifi shape slide
        step = duration / len(touches)
        cur = start
        head = touches[0]
        touch_dict[head[0]][head[1]].append([cur, 1, is_break])
        cur += step
        for i in range(1,4):
            for touch in touches[i]:
                if isinstance(touch[0], list):
                    for item in touch:
                        touch_dict[item[0]][item[1]].append([cur, 1/3, is_break])
                else:
                    touch_dict[touch[0]][touch[1]].append([cur, 2/3, is_break])
            cur += step
    
    def fix(self, pos):
        # The buttons/sensor areas are numbered from 1 to 8 and they are in a ring, so numbner will circulate.
        res = pos%8
        if res == 0:
            res = 8
        return res
    
    # The methods below are all to handle different shape of slides.
    # YOU CAN TELL WHICH TYPE OF SLIDE BY THE METHOD NAME.
    def straight(self, begin, end, shift = 0):
        if begin == 1:
            if end == 3:
                return [ [ 'A', self.fix(1+shift) ], [[ 'A', self.fix(2+shift) ], [ 'B', self.fix(2+shift) ]], [ 'A', self.fix(3+shift) ] ]
            elif end == 4:
                return [ [ 'A', self.fix(1+shift) ], [ 'B', self.fix(2+shift) ], [ 'B', self.fix(3+shift) ], [ 'A', self.fix(4+shift) ] ]
            elif end == 5:
                return [ [ 'A', self.fix(1+shift) ], [ 'B', self.fix(1+shift) ], [ 'C', 1 ],  [ 'B', self.fix(5+shift) ], [ 'A', self.fix(5+shift) ] ]
            elif end == 6:
                return [ [ 'A', self.fix(1+shift) ], [ 'B', self.fix(8+shift) ], [ 'B', self.fix(7+shift) ], [ 'A', self.fix(6+shift) ] ]
            elif end == 7:
                return [ [ 'A', self.fix(1+shift) ], [[ 'A', self.fix(8+shift) ], [ 'B', self.fix(8+shift) ]], [ 'A', self.fix(7+shift) ] ]
        else:
            shift = begin - 1
            return self.straight(1, self.fix(end-shift), shift)
            
    def right_arc_slide(self, begin, end):
        if begin in [1, 2, 7, 8]:
            return self.clockwise(begin, end)
        else:
            return self.counterclockwise(begin, end)

    def left_arc_slide(self, begin, end):
        if begin in [1, 2, 7, 8]:
            return self.counterclockwise(begin, end)
        else:
            return self.clockwise(begin, end)
            
    def clockwise(self, begin, end, zone = 'A', round_flag = True):
        if begin == end and not round_flag:
            return [[zone, begin]]
        else:
            round_flag = False
            #print(begin)
            head = [[zone, begin]]
            return head + self.clockwise( self.fix(begin+1), end, zone, round_flag)

    def counterclockwise(self, begin, end, zone = 'A', round_flag = True):
        if begin == end and not round_flag:
            return [[zone, begin]]
        else:
            round_flag = False
            #print(begin)
            head = [[zone, begin]]
            return head + self.counterclockwise( self.fix(begin-1), end, zone, round_flag)

    def qslide(self, begin, end):
        head = [['A', begin]]
        if begin == end:
            middle = self.clockwise( self.fix(begin+1), self.fix(begin-1), 'B')
        elif begin == self.fix(end-1):
            middle = self.clockwise( self.fix(begin+1), begin, 'B')
        elif begin == self.fix(end-2):
            middle = self.clockwise( self.fix(begin+1), begin, 'B')
            middle.append(['B', self.fix(begin+1)])
        elif begin == self.fix(end-3):
            middle = self.clockwise( self.fix(begin+1), begin, 'B') + self.clockwise( self.fix(begin+1), self.fix(begin+2), 'B')
        elif begin == self.fix(end-4):
            middle = self.clockwise( self.fix(begin+1), self.fix(begin+3), 'B')
        elif begin == self.fix(end-5):
            middle = self.clockwise( self.fix(begin+1), self.fix(begin+4), 'B')
        elif begin == self.fix(end-6):
            middle = self.clockwise( self.fix(begin+1), self.fix(begin+5), 'B')
        elif begin == self.fix(end-7):
            middle = self.clockwise( self.fix(begin+1), self.fix(begin+6), 'B')
        
        tail = [['A', end]]
        return head + middle + tail

    def pslide(self, begin, end):
        result = self.qslide(end, begin)
        result.reverse()
        return result

    def sslide(self, begin, end):
        assert abs(begin-end) == 4, '起始点和结束位置必须是对角'
        head = [['A', begin]]
        middle = self.counterclockwise( self.fix(begin-1), self.fix(begin-2), 'B')
        middle.append(['C', 1])
        middle.append(self.clockwise( self.fix(end-2), self.fix(end-1), 'B'))
        tail = [['A', end]]
        return head + middle + tail

    def zslide(self, begin, end):
        assert abs(begin-end) == 4, '起始点和结束位置必须是对角'
        head = [['A', begin]]
        middle = self.clockwise( self.fix(begin+1), self.fix(begin+2), 'B')
        middle.append(['C', 1])
        middle.append(self.counterclockwise( self.fix(end+2), self.fix(end+1), 'B'))
        tail = [['A', end]]
        return head + middle + tail

    def vslide(self, begin, end):
        return [['A', begin], ['B', begin], ['C', 1], ['B', end], ['A', end]]

    def grandvslide(self, begin, mid, end):
        part1 = self.straight(begin, mid)
        part2 = self.straight(mid, end)
        del part1[-1]
        return part1 + part2

    def qqslide(self, begin, end, shift = 0):
        result = [['A', begin], ['B', begin]]
        result.append(['C', 1])
        result.append(['B', self.fix(begin-3)])
            
        if begin == end:
            result += self.clockwise(self.fix(begin-2), begin)
            
        elif (end-begin)%8 == 1:
            result.append(['A', self.fix(begin-2)])
            result += self.straight(self.fix(begin-1), end)

        elif (end-begin)%8 == 2:
            result += self.clockwise(self.fix(begin-2), self.fix(begin-1))
            result += self.clockwise(begin, self.fix(begin+1), 'B')
            result.append(['A', end])

        elif (end-begin)%8 == 3:
            result += self.clockwise(self.fix(begin-2), self.fix(begin-1))
            result.append(['B', begin])
            result.append([ ['B', self.fix(begin+1)], ['C', 1] ])
            result.append([ ['B', self.fix(begin+2)], ['B', self.fix(begin+3)] ])
            result.append(['A', end])

        elif (end-begin)%8 == 4:
            result += self.clockwise(self.fix(begin-2), self.fix(begin-1))
            #print(result)
            next = self.straight(begin, end)
            del next[0]
            #print(next)
            result += next

        elif (end-begin)%8 == 5:
            result += self.clockwise(self.fix(begin-2), self.fix(begin-1))
            result.append(['B', begin])
            result.append(['C', 1])
            result.append(['B', self.fix(begin-3)])
            result.append(['A', end])

        elif (end-begin)%8 == 6:
            result.append(['A', end])

        elif (end-begin)%8 == 7:
            result += self.clockwise(self.fix(begin-2), self.fix(begin-1))
            
        return result

    def ppslide(self, begin, end):
        mirror_end = self.fix(begin-(end-begin))
        path = self.qqslide(begin, mirror_end)
        #print(path)
        self.mirror(begin, path)
        return path

    def mirror(self, begin, path):
        for item in path:
            if type(item[0]) == str:
                if item[0] != 'C':
                    item[1] = self.fix(begin-(item[1]-begin))
            else:
                self.mirror(begin, item)

    def wifi(self, begin, end):
        left = self.straight(begin, self.fix(end+1))
        middle = self.straight(begin, end)
        right = self.straight(begin, self.fix(end-1))

        left[-1] = [left[-1], ['D', end]]
        right[-1] = [right[-1], ['D', self.fix(end+1)]]
        middle[-2] = [middle[-2], middle[-1]]
        del middle[-1]
        #print(left)
        #print(middle)
        #print(right)
        result = [left[0]]
        for i in range(1,4):
            result.append([left[i], middle[i], right[i]])
        return result

class bpmTotimeConverter():
    # Convert time scale from beats to seconds
    # Uniform unit time everywhere
    def __init__(self, chart):
       #Tap recorder
        self.rating_num = chart['rating_num']
        self.name = chart['name']
        self.chart = chart
        self.bpm = chart['bpm']
        self.tap = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        
        #Hold recorder
        self.hold = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}

        #Touch (hold) recorder
        self.A = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        self.B = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        self.C = {1:[],2:[]}
        self.Ch = []
        self.D = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        self.E = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        self.touch = {'A':self.A, 'B':self.B, 'C':self.C, 'D':self.D, 'E':self.E}
        self.convert()
    
    def output(self):
        for begin in self.tap.keys():
            for item in self.tap[begin]:
                item[0] = round(item[0],4)
            for item in self.hold[begin]:
                item[0] = round(item[0],4)

        for zone in self.touch.keys():
            for begin in self.touch[zone].keys():
                for item in self.touch[zone][begin]:
                    item[0] = round(item[0],4)

        for item in self.Ch:
            item[0] = round(item[0],4)
            
        return {'rating_num' : self.rating_num,
                'name' : self.name,
                'note': {'tap': self.tap,
                         'hold': self.hold,
                         'touch': self.touch,
                         'Ch': self.Ch}
               }
        
    def convert(self):
        # tap
        taps = self.chart['note']['tap']
        for begin in taps.keys():
            for tap in taps[begin]:
                self.tap_convert(tap, int(begin))

        # hold
        holds = self.chart['note']['hold']
        for begin in holds.keys():
            for hold in holds[begin]:
                self.hold_convert(hold, int(begin))

        # touch
        touches = self.chart['note']['touch']
        for zone in touches.keys():
            for begin in touches[zone].keys():
                for touch in touches[zone][begin]:
                    self.touch_convert(touch, zone, int(begin))

        # touch hold
        for Ch in self.chart['note']['Ch']:
            self.Ch_convert(Ch)

        # slide
        slides = self.chart['note']['slide']
        decomposer = slideToSensorConverter()
        sensor_zone = decomposer.convert(slides)
        for zone in sensor_zone.keys():
            for begin in sensor_zone[zone].keys():
                for touch in sensor_zone[zone][begin]:
                    self.touch_convert(touch, zone, int(begin))        
                if len(self.touch[zone][begin]) != 0:
                    self.sort_touch(zone, begin)

    def sort_touch(self, zone, begin):
        # sort everthing in order of time
        arr = np.array(self.touch[zone][begin])
        sorted_indices = np.argsort(arr[:, 0])
        arr[:, 0] = np.round(arr[:, 0], decimals=4)
        arr = list(arr[sorted_indices])
        for i in range(len(arr)):
            arr[i] = list(arr[i])
            arr[i][-1] = int(arr[i][-1])
        self.touch[zone][begin] = arr
        for item in self.touch[zone][begin]:
            item[0] = round(item[0], 4)
            #print(item[0])
    
    def get_interval_ptr(self, time_in_beat):
        # Find which bpm interval a note falls in
        for i in range(len(self.bpm)):
            if time_in_beat <= self.bpm[i][0]:
                return i-1
        return len(self.bpm)-1

    def convert_time(self, time_in_beat, interval_ptr):
        # execute time sscale conversion
        time = 0
        for i in range(interval_ptr):
            num_of_beat = self.bpm[i+1][0] - self.bpm[i+1][0]
            time += num_of_beat / self.bpm[i][1] * 4
        time += (time_in_beat - self.bpm[interval_ptr][0]) / self.bpm[interval_ptr][1] * 4
        return time
    
    def tap_convert(self, tap, begin):
        #tap: [time_in_beat, is_break, is_ex]
        interval_ptr = self.get_interval_ptr(tap[0])
        time = self.convert_time(tap[0], interval_ptr) * 60
        self.tap[begin].append([time] + list(tap[1:]))

    def hold_convert(self, hold, begin):
        #hold: [time_in_beat, duration, is_break, is_ex]
        interval_ptr = self.get_interval_ptr(hold[0])
        time = self.convert_time(hold[0], interval_ptr) * 60
        duration = hold[1] / self.bpm[interval_ptr][1] * 4 * 60
        self.hold[begin].append([time, duration] + list(hold[2:]))

    def touch_convert(self, touch, zone, begin):
        #touch: [time_in_beat, 1, (is_break)]
        interval_ptr = self.get_interval_ptr(touch[0])
        time = self.convert_time(touch[0], interval_ptr) * 60
        info_block = [time] + list(touch[1:])
        if len(info_block) < 3:
            info_block.append(0)
        self.touch[zone][begin].append(info_block)

    def Ch_convert(self, Ch):
        #Ch: [time_in_beat, duration]
        interval_ptr = self.get_interval_ptr(Ch[0])
        time = self.convert_time(Ch[0], interval_ptr) * 60
        duration = Ch[1] / self.bpm[interval_ptr][1] * 4 * 60
        self.Ch.append([time, duration])
        
class noteTokenizer():
    # Input all information needed to form a token
    def __init__(self, data):
        self.data = data
        self.rating_num = data['rating_num']
        self.name = data['name']
        self.tokens = np.empty((18,0))
        self.is_tokenized = False
        self.reference = {'tap':9, 'A':10, 'B':11, 'C':12, 'D':13, 'E':14}
        self.norm_scale = 150
        self.tokenize(True)
        self.sort()

    def sort(self):
        # sort by time
        sort_indices = np.argsort(self.tokens[0, :])
        self.tokens = self.tokens[:, sort_indices]

    def output(self, padding_up_to=None, use_last_time_in_padding=False):
        if padding_up_to != None: # adding 0 vectors until reached desired # of tokens
            padding = padding_up_to - self.tokens.shape[1]
            assert padding > 0, 'Desired number output tokens with padding is smaller than the number of tokens without padding!'
            padding_array = np.zeros((18, padding))
            if use_last_time_in_padding:
                padding_array[0,:] = self.tokens[0,-1]
            self.tokens = np.hstack((self.tokens, padding_array))
        return {'rating_num': self.rating_num,
                'name': self.name,
                'tokens': self.tokens.tolist()}

    
    def tokenize(self, time_normalize=False):
        # token format
        # [time, 1, 2, 3, 4, 5, 6, 7, 8, is_button, A, B, C, D, E, lasting_time, break, ex]
        if self.is_tokenized:
            return;
        self.is_tokenized = True
        chart = self.data['note']
        for begin in chart['tap']:
            for note in chart['tap'][begin]:
                this = self.tap_token(note, int(begin), 'tap', time_normalize)
                self.tokens = np.hstack((self.tokens, this))

        for begin in chart['hold']:
            for note in chart['hold'][begin]:
                this = self.hold_token(note, int(begin), time_normalize)
                self.tokens = np.hstack((self.tokens, this))

        for zone in chart['touch']:
            for begin in chart['touch'][zone]:
                for note in chart['touch'][zone][begin]:
                    this = self.tap_token(note, int(begin), 'tap', time_normalize)
                    self.tokens = np.hstack((self.tokens, this))

        for note in chart['Ch']:
            this = self.Ch_token(note, time_normalize)
            self.tokens = np.hstack((self.tokens, this))

    def tap_token(self, note, begin, type, time_normalize):
        scale = 1
        if time_normalize:
            scale = 1 / self.norm_scale
        token = np.zeros((18,)) 
        token[0] = note[0] * scale
        token[begin] = 1
        token[self.reference[type]] = 1
        token[-2] = note[-2]
        token[-1] = token[-1]
        return token.reshape((18,1))

    def hold_token(self, note, begin, time_normalize):
        scale = 1
        if time_normalize:
            scale = 1 / self.norm_scale
        token = self.tap_token(note, begin, 'tap', time_normalize)
        token[15][0] = note[1] * scale
        return token

    def Ch_token(self, note, time_normalize):
        scale = 1
        if time_normalize:
            scale = 1 / self.norm_scale
        token = np.zeros((18,))
        token[0] = note[0] * scale
        token[1] = 1
        token[12] = 1
        token[15] = note[1] * scale
        return token.reshape((18,1))
