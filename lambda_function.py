from sagemaker.mxnet.model import MXNetPredictor
import os
import io
import boto3
import json
import csv
import string
import sys
import numpy as np
import email

from hashlib import md5


def lambda_handler(event, context):
    s3 = boto3.client("s3")
    vocabulary_length = 9013

    #print("Received event: " + json.dumps(event, indent=2))

    # data = json.loads(json.dumps(event))
    # payload = data['data']
    # print(payload)

    file_obj = event["Records"][0]
    bucketName = str(file_obj["s3"]['bucket']['name'])
    filename = str(file_obj["s3"]['object']['key'])

    data = s3.get_object(Bucket=bucketName, Key=filename)
    print("file has been gotten!")
    msg = email.message_from_bytes(data['Body'].read())
    # print(msg['Subject'])

    RECIPIENT = msg['From']
    SENDER = msg['To']
    EMAIL_RECEIVE_DATE = msg['Date']
    EMAIL_SUBJECT = msg['Subject']

    msgStr = ""
    for a in [k.get_payload() for k in msg.walk() if k.get_content_type() == 'text/plain']:
        msgStr += a
    print(msgStr)
    msgStr = msgStr.replace('\n', ' ').replace('\r', '')
    test_messages = msgStr.split('\n')

    EMAIL_BODY_SHORT = msgStr[0:240]

    mxnet_pred1 = MXNetPredictor(
        'sms-spam-classifier-mxnet-2020-05-10-23-12-11-768')

    #test_messages = ["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop"]
    #test_messages = ["Hello Gaurav this side"]

    print(test_messages)
    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
    encoded_test_messages = vectorize_sequences(
        one_hot_test_messages, vocabulary_length)

    result = mxnet_pred1.predict(encoded_test_messages)
    print(result)

    if(result['predicted_label'][0][0] == 1.0):
        CLASSIFICATION = "SPAM"
    else:
        CLASSIFICATION = "HAM/Not SPAM"

    CLASSIFICATION_CONFIDENCE_SCORE = str(
        round((result['predicted_probability'][0][0] * 100), 2))

    reply_email(RECIPIENT, SENDER, EMAIL_RECEIVE_DATE, EMAIL_SUBJECT,
                EMAIL_BODY_SHORT, CLASSIFICATION, CLASSIFICATION_CONFIDENCE_SCORE)

    #result = json.loads(response['Body'].read().decode())
    # print(result)
    #pred = int(result['predictions'][0]['score'])
    #predicted_label = 'M' if pred == 1 else 'B'

    # return predicted_label


if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans


def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]


def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    """One-hot encodes a text into a list of word indexes of size n.
    This is a wrapper to the `hashing_trick` function using `hash` as the
    hashing function; unicity of word to index mapping non-guaranteed.
    # Arguments
        text: Input text (string).
        n: int. Size of vocabulary.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        List of integers in [1, n]. Each integer encodes a word
        (unicity non-guaranteed).
    """
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    """Converts a text to a sequence of indexes in a fixed-size hashing space.
    # Arguments
        text: Input text (string).
        n: Dimension of the hashing space.
        hash_function: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of integer word indices (unicity non-guaranteed).
    `0` is a reserved index that won't be assigned to any word.
    Two or more words may be assigned to the same index, due to possible
    collisions by the hashing function.
    The [probability](
        https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
    of a collision is in relation to the dimension of the hashing space and
    the number of distinct objects.
    """
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        def hash_function(w): return int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]


def reply_email(RECIPIENT, SENDER, EMAIL_RECEIVE_DATE, EMAIL_SUBJECT, EMAIL_BODY_SHORT, CLASSIFICATION, CLASSIFICATION_CONFIDENCE_SCORE):
    print("reply_email")
    # message = "New User Alert!\nhttp://smart-door-website.s3-website-us-east-1.amazonaws.com/unknown-person.html?uuid={}".format(
    #     row_identifier)
    message = "Test"
    # print(sns_client.publish(PhoneNumber="+19172028241",Message=message))
    SUBJECT = "Re: " + EMAIL_SUBJECT
    BODY_TEXT = ("Amazon SES Test (Python)\r\n"
                 "This email was sent with Amazon SES using the "
                 "AWS SDK for Python (Boto)."
                 )
    BODY_HTML = """<html>
                    <head></head>
                    <body>
                        <p><i>Hello!</i></p>
                        <p><i>We received your email sent at: </i>""" + EMAIL_RECEIVE_DATE + """<i> with the subject: </i>""" + EMAIL_SUBJECT + """</p>
                        <p><i>Here is a 240 character sample of the email body: </i> </p>
                        <p>""" + EMAIL_BODY_SHORT + """</p>
                        <p><i>The email was categorized as </i>""" + CLASSIFICATION + """ <i> with a </i>""" + CLASSIFICATION_CONFIDENCE_SCORE + """% <i>confidence.</i></p>
                    </body>
                    </html>
                """
    CHARSET = "UTF-8"
    client = boto3.client('ses')
    response = client.send_email(
        Destination={
            'ToAddresses': [
                RECIPIENT
            ],
        },
        Message={
            'Body': {
                'Html': {
                    'Charset': CHARSET,
                    'Data': BODY_HTML,
                },
                'Text': {
                    'Charset': CHARSET,
                    'Data': message,
                },
            },
            'Subject': {
                'Charset': CHARSET,
                'Data': SUBJECT,
            },
        },
        Source=SENDER)
    print("ses response", response)
