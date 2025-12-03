import easyocr
reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
result = reader.readtext('D:\\workspace\\vmshareroom\\python_project\\watermarkRemover\\testInput\\test003.jpg')

print(result)
