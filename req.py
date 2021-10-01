import requests

url = 'https://apitest.indusind.com/COUHubComfort/GetBillerCategories'

api_url = url
create_row_data = {'password': 'NormalLogin'}
print(create_row_data)
r = requests.post(url=api_url, json=create_row_data)
print(r.status_code, r.reason, r.text)

# api_url = url + '/video'
# create_row_data = {'link': 'https://drive.google.com/file/d/15EdSWVR-0NegU6pTPHU6oMECQYjzWn0V'}
# print(create_row_data)
# r = requests.post(url=api_url, json=create_row_data)
# print(r.status_code, r.reason, r.text)

api_url = url + '/accuracy'
# create_row_data = {'link': 'https://drive.google.com/file/d/15EdSWVR-0NegU6pTPHU6oMECQYjzWn0V'}
# print(create_row_data)
r = requests.get(url=api_url)
print(r.status_code, r.reason, r.text)

# api_url = url + '/visualization'
# # create_row_data = {'link': 'https://drive.google.com/file/d/15EdSWVR-0NegU6pTPHU6oMECQYjzWn0V'}
# # print(create_row_data)
# r = requests.get(url=api_url)
# print(r.status_code, r.reason, r.text)