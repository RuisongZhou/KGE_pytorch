import os

rootpath = os.path.abspath('../../')+ '/preprocess/'
entity2id_path = rootpath+ 'entity2id.txt'
relation2id_path = rootpath+ 'relation2id.txt'
train_path = rootpath+  'train2id.txt' #'transE_train.txt'
test_path = rootpath+ 'transE_test.txt'
valid_path = rootpath+ 'test2id.txt'  #.txt'

def data2id(pathlist, entity2id_path, relation2id_path):
    entity_dict = {}
    for line in open(entity2id_path, 'r'):
        items = line.strip().split('\t')
        entity_dict[items[0]] = int(items[1])

    relation_dict = {}
    for line in open(relation2id_path, 'r'):
        items = line.strip().split('\t')
        relation_dict[items[0]] = int(items[1])

    for path in pathlist:
        if os.path.exists(path):
            print("[INFO] Prepare dataset:%s" % path)
            outfilePath = rootpath+ 'em'+ path.split('/')[-1]   #rootpath+ 'embedding_'+ path.split('_')[1].split('.')[0]+ '2id.txt'
            if not os.path.exists(outfilePath):
                outfile = open(outfilePath,'w')
                for line in open(path,'r'):
                    items = line.strip().split(' ')
                    head = items[0]
                    tail = items[1]
                    relation = items[2]
                    #if head in entity_dict and tail in entity_dict and relation in relation_dict:
                        #outfile.write('{}\t{}\t{}\n'.format(entity_dict[head], relation_dict[relation], entity_dict[tail]))
                    outfile.write('{}\t{}\t{}\n'.format(head,relation,tail))
                outfile.close()

    print('[INFO] Finish prepare knowledge graph embedding data.')
if __name__ == '__main__':
    data2id([train_path, valid_path], entity2id_path, relation2id_path)
