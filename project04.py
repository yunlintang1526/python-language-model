
import os
import pandas as pd
import numpy as np
import requests
import time
import re

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------


def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    text = requests.get(url).text
    content = text[text.find('*** START'):text.find('*** END')]
    content = re.sub('\*\*\* START.*\*\*\*', '', content)
    content = re.sub('\r\n', '\n', content)
    return content
    
# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of any paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of any paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens in the sequence of words are split 
        apart at 'word boundaries' (see the regex lecture).
        - Tokens should include no whitespace.

    :Example:
    >>> shakespeare_fp = os.path.join('data', 'shakespeare.txt')
    >>> shakespeare = open(shakespeare_fp, encoding='utf-8').read()
    >>> tokens = tokenize(shakespeare)
    >>> tokens[0] == '\x02'
    True
    >>> (tokens[1] == '\x03') and (tokens[-1] == '\x03')
    True
    >>> tokens[10] == 'Shakespeare'
    True
    """
    text = re.sub(r'(.)\n\n', r'\1 \\x03 \\x02', book_string)
    text = re.sub(r'\n\n(w{1})', r' \\x02 \\x03 \1', text)
    text = re.sub('\n\n', '\x03 \x02 ', text)

    if len(text) <= 1:
        text = '\x02 ' + text + ' \x03'
    if text[0] != '\x02':
        text = '\x02 ' + text
    if text[-1] != '\x03':
        text = text + ' \x03'

    text = re.sub(r'\\x03', '\x03', text)
    text = re.sub(r'\\x02', '\x02', text)
    text = re.findall('(\\b\w+\\b|[^\s])', text)
    return text
    
# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """

        idx = pd.Series(tokens).unique()
        return pd.Series([1 / len(idx)]*len(idx), index=idx)
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        try:
            prob=np.prod((pd.Series(words)).apply(lambda x: self.mdl[x]))
            return prob
        except KeyError:
            return 0

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """

        sample_lst = np.random.choice(list(self.mdl.index), M)
        return ' '.join(list(sample_lst))

            
# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        idx = pd.Series(tokens).unique()
        prob = lambda x: tokens.count(x) / len(tokens)
        return pd.Series(list(map(prob, list(idx))), index=idx)
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        try:
            prob = np.prod((pd.Series(words)).apply(lambda x: self.mdl[x]))
            return prob
        except KeyError:
            return 0
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """
        sample_lst = np.random.choice(list(self.mdl.index), M, p=list(self.mdl))
        return ' '.join(list(sample_lst))


# ---------------------------------------------------------------------
# Question #5,6,7,8
# ---------------------------------------------------------------------

class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """

        self.N = N
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', '\\x02')
        >>> out[2]
        ('one', 'two')
        """
        n = self.N
        if len(tokens) == 0:
            return tuple()
        #TODO: will the n be one? if so, change the train method!!!
        if not isinstance(tokens, tuple):
            tokens=tuple(tokens)
        t1 = tuple(('\x02 ' * (n - 1)).split(), )
        t2 = tuple(('\x03 ' * (n - 1)).split(), )

        tok = t1 + tokens + t2
        app = lambda x: tuple([tok[i] for i in range(x, x + n)])
        return list(map(app, list(range(len(tok) - n + 1))))
        
    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (8, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """

        df = pd.DataFrame(columns=['ngram','n1gram','prob'])

        if len(ngrams) < 1:
            return df

        # Create ngram counts C(w_1, ..., w_n)
        if not isinstance(ngrams, tuple):
            ngrams = tuple(ngrams)
        df['ngram'] = ngrams

        # Create n-1 gram counts C(w_1, ..., w_(n-1))
        if self.N - 1 > 0:
            df['n1gram'] = list(map(lambda x: x[:self.N - 1], ngrams))
        else:
            df['n1gram'] = pd.Series()

        # Create the conditional probabilities
        def count(x):
            return df['ngram'].value_counts().to_dict()[x[0]] / df['n1gram'].value_counts().to_dict()[x[1]]
        df['prob'] = df[['ngram', 'n1gram']].apply(count, axis=1)
        
        # Put it all together
        return df
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> bigrams.probability('one two three'.split()) == 0.5
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """
        #TODO: check this
        if not isinstance(words, tuple):
            words=tuple(words)
        # Initialize tokens for words
        app = lambda x: tuple([words[i] for i in range(x, x + self.N)])
        tokens = list(map(app, list(range(len(words) - self.N + 1))))
        df = self.mdl
        # find tokens' prob in ngrams
        find_prob = lambda x: df[df['ngram'] == tokens[x]]['prob'].iloc[0] if sum(df['ngram']==tokens[x]) != 0 else 0
        return np.prod(list(map(find_prob, list(range(len(tokens))))))

    def sample(self, length):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial N-1 START token(s).
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """
        #TODO: the start token??
        # Use a helper function to generate sample tokens of length `length`
        def generate(sentence, count):
            if count == 0:
                return sentence

            token = tuple(sentence.split()[-(self.N - 1):])
            df_prob = df[df['n1gram'] == token]
            if df_prob.shape[0] == 0:
                sentence = sentence + ' ' + '\x03'
            else:
                sentence = sentence + ' ' + (np.random.choice(df_prob['ngram'], 1, p=df_prob['prob']))[0][-1]
            count -= 1
            return generate(sentence, count)
        
        # Tranform the tokens to strings
        df = self.mdl
        first = ('\x02',) * (self.N - 1)
        df_prob = df[df['n1gram'] == first]
        s = ' '.join(first) + ' ' + (np.random.choice(df_prob['ngram'], 1, p=df_prob['prob']))[0][-1]

        return generate(s, length-1)

# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------
    

def predict_next(lm, tokens):
    """
    predict_next that takes in an instantiated NGramLM object 
    and a list of tokens, and predicts the most likely token to 
    follow the list of tokens in the input (according to NGramLM).

    :Example:
    >>> tokens = tuple('\x02 one two three one four \x03'.split())
    >>> bigrams = NGramLM(2, tokens)
    >>> predict_next(bigrams, ('one', 'two')) == 'three'
    True
    """

    if not isinstance(tokens, tuple):
        tokens = tuple(tokens)

    n1gram = tokens[-(lm.N-1):]
    df = lm.mdl
    sort = df[df['n1gram']==n1gram].sort_values(by='prob')
    if sort.shape[0] == 0:
        return '\x03'
    return (sort['ngram'].iloc[0])[-1]

# ---------------------------------------------------------------------
# Question # 10
# ---------------------------------------------------------------------
    

def evaluate_models(tokens):
    """
    Extra-Credit (see notebook)
    """

    return ...

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_book'],
    'q02': ['tokenize'],
    'q03': ['UniformLM'],
    'q04': ['UnigramLM'],
    'q05': ['NGramLM'],
    'q09': ['predict_next'],
    'q10': ['evaluate_models']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
