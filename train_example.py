import pickle 
from models.hostlevel import NodeRNN

with open('nodes201.pkl', 'rb') as f:
    nodes = pickle.load(f)


f_rnn = NodeRNN(nodes.file_dim, 16, 8)
t, x, i = nodes.sample_feat('files')

embeddings = f_rnn(t,x)
print(embeddings)
# Repeat for each feature, and cat together(?)
# Train against GAN that generates real looking embeds(?)