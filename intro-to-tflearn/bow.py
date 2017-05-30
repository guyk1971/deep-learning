def bag_of_words(text):
    bow=dict()
    for w in text.split(' '):
        if w in bow.keys():
            bow[w]+=1
        else:
            bow[w]=1
    return bow

test_text = 'the quick brown fox jumps over the lazy dog'

print(bag_of_words(test_text))