import numpy as np
import pandas as pd
import re
import string
from underthesea import word_tokenize, text_normalize


VIET_TAT = {
    " sp " : "sản phẩm",
    " dc ": "được",
    " k ": "không",
    " ko ": "không",
    " r ": "rồi",
    " oke ": "ok",
    " okie ": "ok",
    " okey ": "ok",
    " nt ": "nhắn tin",
    " bt ": "biết",
    " bít ": "biết",
    " ae ": "anh em",
    " nx ": "nữa",
    " nv ": "nhân viên",
    " tv ": "tư vấn",
    " mn ": "mọi người"
}

replace_list = {
        'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ', 'òe': 'oè', 'óe': 'oé','ỏe': 'oẻ',
        'õe': 'oẽ', 'ọe': 'oẹ', 'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ','ụy': 'uỵ', 'uả': 'ủa',
        'ả': 'ả', 'ố': 'ố', 'u´': 'ố','ỗ': 'ỗ', 'ồ': 'ồ', 'ổ': 'ổ', 'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ',
        'ầ': 'ầ', 'ỏ': 'ỏ', 'ề': 'ề','ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 'ế': 'ế', 'ở': 'ở', 'ỉ': 'ỉ',
        'ẻ': 'ẻ', 'àk': u' à ','aˋ': 'à', 'iˋ': 'ì', 'ă´': 'ắ','ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ', 'a´': 'á',
        #Quy các icon về 2 loại emoj: Tích cực hoặc tiêu cực
        "👹": " negative ", "👻": " positive ", "💃": " positive ",'🤙': ' positive ', '👍': ' positive ',
        "💄": " positive ", "💎": " positive ", "💩": " positive ","😕": " negative ", "😱": " negative ", "😸": " positive ",
        "😾": " negative ", "🚫": " negative ",  "🤬": " negative ","🧚": " positive ", "🧡": " positive ",'🐶':' positive ',
        '👎': ' negative ', '😣': ' negative ','✨': ' positive ', '❣': ' positive ','☀': ' positive ',
        '♥': ' positive ', '🤩': ' positive ', 'like': ' positive ', '💌': ' positive ',
        '🤣': ' positive ', '🖤': ' positive ', '🤤': ' positive ', ':(': ' negative ', '😢': ' negative ',
        '❤': ' positive ', '😍': ' positive ', '😘': ' positive ', '😪': ' negative ', '😊': ' positive ',
        '?': ' ? ', '😁': ' positive ', '💖': ' positive ', '😟': ' negative ', '😭': ' negative ',
        '💯': ' positive ', '💗': ' positive ', '♡': ' positive ', '💜': ' positive ', '🤗': ' positive ',
        '^^': ' positive ', '😨': ' negative ', '☺': ' positive ', '💋': ' positive ', '👌': ' positive ',
        '😖': ' negative ', '😀': ' positive ', ':((': ' negative ', '😡': ' negative ', '😠': ' negative ',
        '😒': ' negative ', '🙂': ' positive ', '😏': ' negative ', '😝': ' positive ', '😄': ' positive ',
        '😙': ' positive ', '😤': ' negative ', '😎': ' positive ', '😆': ' positive ', '💚': ' positive ',
        '✌': ' positive ', '💕': ' positive ', '😞': ' negative ', '😓': ' negative ', '️🆗️': ' positive ',
        '😉': ' positive ', '😂': ' positive ', ':v': '  positive ', '=))': '  positive ', '😋': ' positive ', "🙆": ' positive ', "🤍": ' positive ', "🥰": ' positive ',
        '💓': ' positive ', '😐': ' negative ', ':3': ' positive ', '😫': ' negative ', '😥': ' negative ', '😅': ' positive ',
        '😃': ' positive ', '😬': ' negative ', '😌': ' positive ', '💛': ' positive ', '🤝': ' positive ', '🎈': ' positive ',
        '😗': ' positive ', '🤔': ' negative ', '😑': ' negative ', '🔥': ' negative ', '🙏': ' negative ',
        '🆗': ' positive ', '😻': ' positive ', '💙': ' positive ', '💟': ' positive ',
        '😚': ' positive ', '❌': ' negative ', '👏': ' positive ', ';)': ' positive ', '<3': ' positive ',
        '🌝': ' positive ',  '🌷': ' positive ', '🌸': ' positive ', '🌺': ' positive ',
        '🌼': ' positive ', '🍓': ' positive ', '🐅': ' positive ', '🐾': ' positive ', '👉': ' positive ',
        '💐': ' positive ', '💞': ' positive ', '💥': ' positive ', '💪': ' positive ',
        '💰': ' positive ',  '😇': ' positive ', '😛': ' positive ', '😜': ' positive ',
        '🙃': ' positive ', '🤑': ' positive ', '🤪': ' positive ','☹': ' negative ',  '💀': ' negative ',
        '😔': ' negative ', '😧': ' negative ', '😩': ' negative ', '😰': ' negative ', '😳': ' negative ', "🥲": ' negative ', '🙄': ' negative ', '🤦': ' negative ',
        '😵': ' negative ', '😶': ' negative ', '🙁': ' negative ', "⏸": " product ", "★": ' star ', '♀': " ", '😅': " positive ", " 👋 ": " ", " 🏻 ": " product ", "🏿": " product ", "😽": " positve ",
        "😬" : ' negative ', '🤷': ' negative ', '😌': ' poisitive ', "😿": ' negative ', '✓': " positive ", '☆': ' star ', 'ಠ｣ಠ༎ຶ‿༎ຶ': ' ', '🐵': ' negative ',
        "👂" : ' ', "😴": ' negative ', '👋': ' ', '\xa0': ' ', '🏻': ' product ', '🏻': ' product ',
        #Chuẩn hóa 1 số sentiment words/English words
        ':))': '  positive ', ':)': ' positive ', 'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ', "cmn": "cm",
        ' okey ': ' ok ', ' ôkê ': ' ok ', ' oki ': ' ok ', ' oke ':  ' ok ',' okay ':' ok ',' okê ':' ok ',
        ' tks ': u' cám ơn ', ' thks ': u' cám ơn ', ' thanks ': u' cám ơn ', ' ths ': u' cám ơn ', ' thank ': u' cám ơn ',
        '⭐': 'star ', '*': 'star ', '🌟': 'star ', '🎉': u' positive ', "bik": " bị ",
        ' kg ': u' không ','not': u' không ', u' kg ': u' không ', ' k ': u' không ',' kh ':u' không ',' kô ':u' không ','hok':u' không ',' kp ': u' không phải ',u' kô ': u' không ', '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
        'he he': ' positive ','hehe': ' positive ','hihi': ' positive ', ' haha': ' positive ', 'hjhj': ' positive ',
        ' lol ': ' negative ',' cc ': ' negative ','cute': u' dễ thương ',' huhu ': ' negative ', ' vs ': u' với ', ' wa ': ' quá ', 'wá': u' quá', ' j ': u' gì ', '“': ' ',
        ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', ' dc ': u' được ', 'đk': u' được ',
        ' đc ': u' được ',' authentic ': u' chính hãng ',u' aut ': u' chính hãng ', u' auth ': u' chính hãng ', ' thick ': u' positive ', ' store ': u' cửa hàng ',
        ' shop ': u' cửa hàng ', ' sp ': u' sản phẩm ', ' gud ': u' tốt ',' god ': u' tốt ',' wel done ':' tốt ', ' good ': u' tốt ', ' gút ': u' tốt ',
        ' sấu ': u' xấu ',' gut ': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', ' perfect ': 'rất tốt',
        'time': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',
        ' ể ': 'ể', ' product ': 'sản phẩm', ' quality ': 'chất lượng', ' excelent ': 'hoàn hảo', ' bad ': ' tệ ',' fresh ': ' tươi ','sad': ' tệ ',
        ' date ': u' hạn sử dụng ', ' hsd ': u' hạn sử dụng ',' quickly ': u' nhanh ', ' quick ': u' nhanh ',' fast ': u' nhanh ',' delivery ': u' giao hàng ',u' síp ': u' giao hàng ',
        ' beautiful ': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shope ': u' cửa hàng ',u' order ': u' đặt hàng ',
        ' chất lg ': u' chất lượng ',u' sd ': u' sử dụng ',u' dt ': u' điện thoại ',u' nt ': u' nhắn tin ',u' tl ': u' trả lời ',u' sài ': u' xài ',u' bjo ':u' bao giờ ',
        ' thik ': u' thích ',u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',u'quả ng ':u' quảng  ',
        ' dep ': u' đẹp ',u' xau ': u' xấu ',' delicious ': u' ngon ', u' hàg ': u' hàng ', u' qủa ': u' quả ',
        ' iu ': u' yêu ','fake': u' giả mạo ', ' trl ': 'trả lời', '><': u' positive ',
        ' por ': u' tệ ',' poor ': u' tệ ', ' ib ':u' nhắn tin ', ' rep ':u' trả lời ',u' fback ':' feedback ',' fedback ':' feedback ',
        #dưới 3* quy về 1*, trên 3* quy về 5*
        '6 sao': ' 5star ','6 star': ' 5star ', '5star': ' 5star ','5 sao': ' 5star ','5sao': ' 5star ',
        'starstarstarstarstar': ' 5star ', '1 sao': ' 1star ', '1sao': ' 1star ','2 sao':' 1star ','2sao':' 1star ',
        '2 starstar':' 1star ','1star': ' 1star ', '0 sao': ' 1star ', '0star': ' 1star ',}

def remove_duplicate_ending_letters(word):

    """A function to remove duplicated ending letters in a word (For example: from "okkk" to "ok")
    Parameters
    ----------
    word : str
    A word with duplicated ending letters 

    Returns
    ----------
    str
    A word with no duplicated ending letter
    """
    pattern = r"\b(\w+)(.)\2{1}\b"
    replaced_word = re.sub(pattern, r"\1\2", word)
    while replaced_word != word:
        word = replaced_word
        pattern = r"\b(\w+)(.)\2{1}\b"
        replaced_word = re.sub(pattern, r"\1\2", replaced_word)
    return replaced_word

def expand_contractions(sentence, contraction_mapping):

    """Expand words written in short hands or teen code (For example: "Câu này dc dùng làm ví dụ" to "Câu này được dùng làm ví dụ") 
    Parameters
    ----------
    sentence : str
    A sentence with contracted words

    contraction_mapping: dict
    Contains the words needs replacement as key and the replacement as values

    Returns
    ----------
    str
    Sentence with no contractions
    """

    sentence = " " + sentence + " "
    contraction_pattern = re.compile("({})".format("|".join(contraction_mapping.keys())), flags= re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[1]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())
        expanded_contraction = " " + first_char + expanded_contraction[1:] + " "
        return expanded_contraction
    expanded_sentence = contraction_pattern.sub(expand_match, sentence)
    while sentence != expanded_sentence:
        sentence = expanded_sentence
        expanded_sentence = contraction_pattern.sub(expand_match, sentence)
    return expanded_sentence.strip()

def remove_all_numbers(sentence):

    """Remove all numbers in a sentence
    Parameters
    ----------
    sentence : str
    A sentence with numbers

    Returns
    ----------
    str
    Sentence with no number
    """
    
    return re.sub(r'[0-9]', '', sentence)

def replace_sent(text, replace_list):

    """Remove elements in a text according to the replacement dictionary
    Parameters
    ----------
    text : str
    Text that needs replacement

    Returns
    ----------
    str
    Text with all elements in the replacement dictionary replaced
    """

    for k, v in replace_list.items():
        text = text.replace(k, v)
    return text

def remove_punctuation(text):
    """Remove all punctuations in a text
    Parameters
    ----------
    text : str
    Text that has punctuations 

    Returns
    ----------
    str
    Text with no punctuation
    """
    return text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))

def remove_all_tag(sentence):
    """Remove tags added from reading files
    Parameters
    ----------
    text : str
    Text that has tags

    Returns
    ----------
    str
    Text with no tags
    """
    return re.sub(r'\n|\t|\r', ' ', sentence)

def cleaning(sentence):
    """ A function to process a sentence in following order:
    1. lowercase all character
    2. normalize all vietnamese words
    3. remove all numbers
    4. remove punctuations
    5. expanding contractions
    6. replace special words and symbols
    7. remove duplicated ending letters in words
    8. remove tags
    ----------
    sentence : str
    input string

    Returns
    ----------
    str
    processed string
    """
    sentence = sentence.lower()
    sentence = text_normalize(sentence)
    sentence = remove_all_numbers(sentence)
    sentence = remove_punctuation(sentence) # delete punctuation
    sentence = expand_contractions(sentence, contraction_mapping = VIET_TAT) # fix shorthand
    sentence = replace_sent(sentence, replace_list)
    sentence = remove_duplicate_ending_letters(sentence)
    sentence =  remove_all_tag(sentence)
    return sentence