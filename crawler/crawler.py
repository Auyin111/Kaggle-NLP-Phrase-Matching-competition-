import re
import chromedriver_binary
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import time

chromedriver_binary
from selenium.webdriver.common.keys import Keys

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# for mac envir only atm, need to install chromium driver
options = Options()
options.binary_location = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
options.headless = True
driver = webdriver.Chrome(options=options)

all_groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']
all_context = set()


def clean_context(context):
    return context[:3]


def detect_mentioned_context(d):
    return set(filter(lambda c: c in d, all_context))


def remove_content_in_bracket(r):
    return re.sub("[\(\[\{].*?[\)\]\}]", "", r)


def clean_descriptions(d):
    return remove_content_in_bracket(d).lower()


# def get_all_context_group():
#     for group in all_groups:
#         url = f'https://www.uspto.gov/web/patents/classification/cpc/html/cpc-{group}.html'
#         driver.get(url)
#         contexts = [clean_context(s.text) for s in driver.find_elements_by_class_name("symbol")]
#         for c in contexts:
#             if len(c) > 1:
#                 all_context.add(c)

# hardcode here, result generated from above commented code
all_context = set({'B22', 'F22', 'F41', 'B25', 'D01', 'C06', 'E99', 'C14', 'A43', 'B31', 'H02', 'F28', 'C09', 'B21',
                   'B33', 'B81', 'B65', 'C99', 'A42', 'B67', 'A99', 'C40', 'B61', 'E04', 'D05', 'B03', 'B07', 'A63',
                   'D99', 'B42', 'B66', 'D10', 'G21', 'C10', 'C23', 'A24', 'F04', 'C30', 'Y04', 'C21', 'G99', 'D07',
                   'B09', 'B29', 'C08', 'B05', 'B44', 'B62', 'B68', 'F15', 'G08', 'F24', 'C07', 'G09', 'A45', 'F01',
                   'C01', 'B01', 'C13', 'E03', 'C12', 'H04', 'H05', 'G16', 'D04', 'B04', 'B30', 'A44', 'Y10', 'F05',
                   'A61', 'B63', 'D02', 'B99', 'D06', 'D03', 'C25', 'C22', 'F17', 'B43', 'G07', 'E05', 'A41', 'G03',
                   'A46', 'B23', 'G01', 'F03', 'F26', 'F16', 'C04', 'G12', 'A21', 'E02', 'Y02', 'B82', 'B27', 'B24',
                   'E01', 'B60', 'H01', 'B32', 'G05', 'B02', 'F02', 'A22', 'A23', 'B06', 'A62', 'F25', 'F27', 'A01',
                   'F42', 'G04', 'H03', 'C02', 'G10', 'B28', 'F23', 'G02', 'F21', 'D21', 'A47', 'B64', 'E21', 'C11',
                   'C05', 'G11', 'B41', 'H99', 'E06', 'F99', 'B26', 'C03', 'G06', 'B08'})

result = pd.DataFrame(columns=['context', 'description', 'mentioned_groups'])

for group in all_groups:
    url = f'https://www.uspto.gov/web/patents/classification/cpc/html/cpc-{group}.html'

    driver.get(url)
    html = driver.find_element_by_tag_name('body')

    contexts = [s.text for s in driver.find_elements_by_xpath("//span[@class='alink']//span[@class='alink']")]
    descriptions = [s.text for s in driver.find_elements_by_xpath("//td[not(@span=2)]//div[@class='class-title']")]

    for i, (c, d) in enumerate(zip(contexts, descriptions)):
        if len(c) == 1:  # Ignore A, B, C... group title, as already included before
            continue
        mentioned_group = detect_mentioned_context(d)
        if len(c) <= 3:
            result = result.append({'context': clean_context(c), 'description': clean_descriptions(d), 'mentioned_groups': mentioned_group},
                                    ignore_index=True)
        else:
            result = result.append({'context': clean_context(c), 'description': "", 'mentioned_groups': mentioned_group},
                                    ignore_index=True)

result = result.groupby('context').agg({'description': ''.join, 'mentioned_groups': lambda x: set.union(*x)})

# Clean up the mention_groups column
result["mentioned_groups"] = result["mentioned_groups"].apply(lambda x: x if x != set() else "").apply(lambda s: re.sub(r"[{}']", "", str(s))).apply(lambda c: re.sub(r"[,]", ";", c))

result.to_csv('result.csv')

# Clean up
driver.close()
print('done')
