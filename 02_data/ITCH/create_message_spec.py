#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import pandas as pd

df = pd.read_excel('message_types.xlsx', sheetname='messages', encoding='latin1').sort_values('id').drop('id', axis=1)

df.columns = [c.lower().strip() for c in df.columns]
df.value = df.value.str.strip()
df.name = df.name.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_').str.replace('/', '_')
df.notes = df.notes.str.strip()
df['message_type'] = df.loc[df.name == 'message_type', 'value']


def get_message_labels():
    messages = df.loc[:, ['message_type', 'notes']].dropna().rename(columns={'notes': 'name'})
    messages.name = messages.name.str.lower().str.replace('message', '')
    messages.name = messages.name.str.replace('.', '').str.strip().str.replace(' ', '_')
    messages.to_csv('message_labels.csv', index=False)


df.message_type = df.message_type.ffill()
df = df[df.name != 'message_type']
df.value = df.value.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')


# Check field count
def check_field_count(df):
    message_size = pd.read_excel('message_types.xlsx', sheetname='size', index_col=0)
    message_size['check'] = df.groupby('message_type').size()
    assert message_size['size'].equals(message_size.check), 'field count does not match template'


def check_field_specs():
    messages = df.groupby('message_type')
    for t, message in messages:
        print(message.offset.add(message.length).shift().fillna(0).astype(int).equals(message.offset))


df[['message_type', 'name', 'value', 'length', 'offset', 'notes', 'coding']].to_csv('message_types.csv', index=False)
