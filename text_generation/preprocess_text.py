import re
from configs import configs

def lowercase(text):
    """Converts to lowercase"""
    new_text = []
    for line in text:
        new_text.append(line.lower())
    return new_text


def rem_measurement(text):
    """Removes measurements such as 2 cm, 1-6 mm, 5.6 cm"""
    new_text = []
    for line in text:
        temp = re.sub('\d*?-?.?\d* [cm]m', '', line)
        new_text.append(temp)
    return new_text


def rem_dcolon(text):
    """Removes d: from the text"""
    new_text = []
    for line in text:
        temp = re.sub('d:', '', line)
        new_text.append(temp)
    return new_text


def rem_irrelevant_strings(text):
    """Removes irrevelant substrings from text"""
    new_text = []
    irrelevant = [
        'as compared to the previous radiograph',
        'no previous images',
        'dr.'
    ]
    for line in text:
        temp = line
        for ir in irrelevant:
            temp = temp.replace(ir, '')
        new_text.append(temp)

    return new_text


def rem_number_listing(text):
    """Removes number listing like 1. 2. 3."""
    new_text = []
    for line in text:
        temp = re.sub('\d.', '', line)
        new_text.append(temp)
    return new_text


def rem_comparison(text):
    """Removes comparison related string"""
    comparison_regex = ['^(as)?\s?compared to(.*?)[\.\,]',
                        '^comparison is made(.*?)[\.\,]',
                        '^(in)?\s?comparison with(.*?)\,']
    new_text = []
    for line in text:
        temp = line
        for cmp in comparison_regex:
            temp = re.sub(cmp, '', temp)
        new_text.append(temp)
    return new_text


def decontractions(text):
    """Performs decontractions in the doc"""
    new_text = []
    for phrase in text:
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"couldn\'t", "could not", phrase)
        phrase = re.sub(r"shouldn\'t", "should not", phrase)
        phrase = re.sub(r"wouldn\'t", "would not", phrase)
        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        phrase = re.sub(r"\*+", "abuse", phrase)
        new_text.append(phrase)

    return new_text


def rem_time(text):
    """Removes time"""
    new_text = []
    for line in text:
        temp = re.sub(r"\d{1,2}:\d{2}", '', line)
        temp = temp.replace('am', '')
        temp = temp.replace('a.m.', '')
        temp = temp.replace('pm', '')
        temp = temp.replace('p.m.', '')
        temp = temp.replace('m.', '')  # Leftover from rem_comparison
        new_text.append(temp)
    return new_text


def rem_punctuations(text):
    """Removes punctuations"""
    punctuations = """=!()-[]{};:'"\,<>/?@#$%^&*~_+`|"""  # full stop is not removed
    new_text = []
    for line in text:
        for char in line:
            if char in punctuations:
                line = line.replace(char, "")
        new_text.append(' '.join(e for e in line.split()))
    return new_text


def multiple_fullstops(text):
    """ Removes multiple full stops from the text"""
    new_text = []
    for line in text:
        new_text.append(re.sub(r'\.\.+', '.', line))
    return new_text


def fullstops(text):
    new_text = []
    for line in text:
        new_text.append(re.sub('\.', ' .', line))
    return new_text


def multiple_spaces(text):
    # new_text = []
    # for line in text:
    #     new_text.append(' '.join(e for e in line.split()))
    new_text = [re.sub(' {2,}', ' ', x).strip() for x in text]
    return new_text


def multiple_x(text):
    # new_text = []
    # for line in text:
    #     new_text.append(' '.join(e for e in line.split()))
    new_text = [re.sub(' x{2,}', '', x).strip() for x in text]
    new_text = [re.sub('x{2,}', '', x).strip() for x in text]
    return new_text


def separting_startg_words(text):
    new_text = []
    for line in text:
        temp = []
        words = line.split()
        for i in words:
            if i.startswith('.') == False:
                temp.append(i)
            else:
                w = i.replace('.', '. ')
                temp.append(w)
        new_text.append(' '.join(e for e in temp))
    return new_text


def rem_apostrophes(text):
    new_text = []
    for line in text:
        new_text.append(re.sub("'", '', line))
    return new_text


def rem_beginning_fullstop(text):
    new_text = []
    for line in text:
        new_text.append(re.sub('^\.', '', line))
    return new_text


def text_preprocessing(text):
    """
    Combines all the preprocess functions
    """

    new_text = lowercase(text)
    new_text = rem_measurement(new_text)
    new_text = rem_dcolon(new_text)
    new_text = rem_irrelevant_strings(new_text)
    new_text = rem_number_listing(new_text)
    new_text = rem_comparison(new_text)
    new_text = rem_time(new_text)

    new_text = decontractions(new_text)
    new_text = rem_punctuations(new_text)
    new_text = multiple_fullstops(new_text)
    new_text = multiple_x(new_text)
    new_text = fullstops(new_text)
    new_text = separting_startg_words(new_text)
    new_text = rem_apostrophes(new_text)
    new_text = multiple_spaces(new_text)
    new_text = rem_beginning_fullstop(new_text)
    return [text.strip() for text in new_text]


def remove_empty(inp):
    """
    There are some missing reports in the dataset
    So we remove it here
    """

    df = inp.copy()
    length = df['report'].str.len()
    res = df[length > 1].reset_index(drop=True)

    return res

def add_start_stop(text) -> str:
    return configs['START_TOK'] + ' ' + text.str.strip() + ' ' + configs['STOP_TOK']


def preprocess_text(df):
    """
    Preprocess 'report' column of df according to preprocess_text.py
    """

    cleaned_df = df.copy()
    cleaned_df['report'] = text_preprocessing(df['report'])

    # Remove empty
    cleaned_df = remove_empty(cleaned_df)
    cleaned_df['report'] = add_start_stop(cleaned_df['report'])

    # No manipulation here, just see what's the min and max length
    lengths = cleaned_df['report'].apply(lambda x: x.split()).str.len()
    print(
        f"Max report length = {lengths.max()}, min report length = {lengths.min()}")

    return cleaned_df

if __name__ == '__main__':
    t = [
        'hello xx he xxx hellooo xxxx .',
        'xx xxxx xxxxx ee',
    ]
    print(multiple_x(t))