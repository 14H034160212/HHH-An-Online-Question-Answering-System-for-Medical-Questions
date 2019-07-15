

import os
import ahocorasick

class QuestionClassifier:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        #　特征词路径
        self.disease_path = os.path.join(cur_dir, 'dict/disease.txt')
        self.department_path = os.path.join(cur_dir, 'dict/department.txt')
        self.symptom_path = os.path.join(cur_dir, 'dict/symptoms.txt')
        self.deny_path = os.path.join(cur_dir, 'dict/deny.txt')
        # 加载特征词
        self.disease_wds= [i.strip() for i in open(self.disease_path,'r', encoding='gbk') if i.strip()]
        self.department_wds= [i.strip() for i in open(self.department_path,'r', encoding='gbk') if i.strip()]
        self.symptom_wds= [i.strip() for i in open(self.symptom_path,'r', encoding='utf-8') if i.strip()]
        self.region_words = set(self.department_wds + self.disease_wds + self.symptom_wds)
        self.deny_words = [i.strip() for i in open(self.deny_path,'r', encoding='gbk') if i.strip()]
        # 构造领域actree
        self.region_tree = self.build_actree(list(self.region_words))
        # 构建词典
        self.wdtype_dict = self.build_wdtype_dict()
        # 问句疑问词
        self.symptom_qwds = ['symptom', 'characterization', 'phenomenon']
        self.cause_qwds = ['reason','cause']
        self.acompany_qwds = ['complication', 'concurrent', 'occur','happen together', 'occur together', 'appear together', 'together', 'accompany', 'follow', 'coexist']
        self.prevent_qwds = ['prevention', 'prevent', 'resist', 'guard', 'against','escape','avoid',
                             'how can I not', 
                             'how not to', 'why not', 'how to prevent']
        self.lasttime_qwds = ['cycle', 'time','day','year','hour','days','years','hours','how long', 'how much time', 'a few days', 'how many years', 'how many days', 'how many hours', 'a few hours', 'a few years']
        self.cureway_qwds = ['treat','heal','cure','how to treat', 'how to heal', 'how to cure', 'treatment', 'therapy']
        self.cureprob_qwds = ['how big is the hope of cure', 'hope','probability', 'possibility', 'percentage', 'proportion']
        self.easyget_qwds = ['susceptible population', 'susceptible','crowd','easy to infect', 'who', 'which people', 'infection', 'infect']
        self.belong_qwds = ['what belongs to', 'belong', 'belongs','section','what section', 'department']
        self.cure_qwds = ['what to treat', 'indication', 'what is the use', 'benefit', 'usefulness']

        print('model init finished ......')

        return

    '''分类主函数'''
    def classify(self, question):
        data = {}
        question2 = question.lower()
        medical_dict = self.check_medical(question2)
        if not medical_dict:
            return {}
        data['args'] = medical_dict
        #收集问句当中所涉及到的实体类型
        types = []
        for type_ in medical_dict.values():
            types += type_
#            print('type_ is '+type_)
        question_type = 'others'

        question_types = []

        # 症状
        if self.check_words(self.symptom_qwds, question2) and ('disease' in types):
            question_type = 'disease_symptom'
            question_types.append(question_type)

        if self.check_words(self.symptom_qwds, question2) and ('symptom' in types):
            question_type = 'symptom_disease'
            question_types.append(question_type)

        # 原因
        if self.check_words(self.cause_qwds, question2) and ('disease' in types):
            question_type = 'disease_cause'
            question_types.append(question_type)
        # 并发症
        if self.check_words(self.acompany_qwds, question2) and ('disease' in types):
            question_type = 'disease_acompany'
            question_types.append(question_type)


        #　症状防御
        if self.check_words(self.prevent_qwds, question2) and 'disease' in types:
            question_type = 'disease_prevent'
            question_types.append(question_type)

        # 疾病医疗周期
        if self.check_words(self.lasttime_qwds, question2) and 'disease' in types:
            question_type = 'disease_lasttime'
            question_types.append(question_type)

        # 疾病治疗方式
        if self.check_words(self.cureway_qwds, question2) and 'disease' in types:
            question_type = 'disease_cureway'
            question_types.append(question_type)

        # 疾病治愈可能性
        if self.check_words(self.cureprob_qwds, question2) and 'disease' in types:
            question_type = 'disease_cureprob'
            question_types.append(question_type)

        # 疾病易感染人群
        if self.check_words(self.easyget_qwds, question2) and 'disease' in types :
            question_type = 'disease_easyget'
            question_types.append(question_type)

        # 若没有查到相关的外部查询信息，那么则将该疾病的描述信息返回
        if question_types == [] and 'disease' in types:
            question_types = ['disease_desc']

        # 若没有查到相关的外部查询信息，那么则将该疾病的描述信息返回
        if question_types == [] and 'symptom' in types:
            question_types = ['symptom_disease']

        # 将多个分类结果进行合并处理，组装成一个字典
        data['question_types'] = question_types

        return data

    '''构造词对应的类型'''
    def build_wdtype_dict(self):
        wd_dict = dict()
        for wd in self.region_words:
            wd_dict[wd] = []
            if wd in self.disease_wds:
                wd_dict[wd].append('disease')
            if wd in self.department_wds:
                wd_dict[wd].append('department')
            if wd in self.symptom_wds:
                wd_dict[wd].append('symptom')
        return wd_dict

    '''构造actree，加速过滤'''
    def build_actree(self, wordlist):
        actree = ahocorasick.Automaton()
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))
#            print ('actree construction '+word)
        actree.make_automaton()
        return actree

    '''问句过滤'''
    def check_medical(self, question):
        region_wds = []
        for i in self.region_tree.iter(question):
            wd = i[1][1]
            region_wds.append(wd)
#            print ('check_medical '+wd)
        stop_wds = []
        for wd1 in region_wds:
            for wd2 in region_wds:
                if wd1 in wd2 and wd1 != wd2:
                    stop_wds.append(wd1)
        final_wds = [i for i in region_wds if i not in stop_wds]
        final_dict = {i:self.wdtype_dict.get(i) for i in final_wds}
#        print ('final_dict is '+final_dict)
        return final_dict

    '''基于特征词进行分类'''
    def check_words(self, wds, sent):
        for wd in wds:
            if wd in sent:
                return True
        return False


if __name__ == '__main__':
    handler = QuestionClassifier()
    while 1:
        question = input('input an question:')
        data = handler.classify(question)
        print(data)