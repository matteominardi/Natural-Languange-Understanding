def desc_statistics(words, sents):
    word_lens = [len(word) for word in words]
    sent_lens = [len(sent) for sent in sents]
    chars_in_sents = [len(''.join(sent)) 
                      if type(sent) == 'str' 
                      else len(''.join(f"{sent[:5]}")) for sent in sents]
    
    word_per_sent = round(sum(sent_lens) / len(sents))
    char_per_word = round(sum(word_lens) / len(words))
    char_per_sent = round(sum(chars_in_sents) / len(sents))
    
    longest_sentence = max(sent_lens)
    longest_word = max(word_lens)
    
    return word_per_sent, char_per_word, char_per_sent, longest_sentence, longest_word


def print_desc_statistics(words, sents):
    word_per_sent, char_per_word, char_per_sent, longest_sent, longest_word = desc_statistics(words, sents)

    print("Number of tokens:", len(words))
    print("Number of sents:", len(sents))
    print('Word per sentence', word_per_sent)
    print('Char per word', char_per_word)
    print('Char per sentence', char_per_sent)
    print('Longest word', longest_word)
    print('Longest sentence', longest_sent)


def nbest(d, n=1):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])