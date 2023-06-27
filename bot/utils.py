import re

def pdf2txt(pdf_file):
    import pdfplumber
    file_handle = open("data/《中华人民共和国民法典》.txt", mode='w', encoding='utf-8')
    with pdfplumber.open(pdf_file) as pdf:
        main_content = ""
        for page in pdf.pages:
            text = page.extract_text()
            main_content += remove_numeric_string(text)
        file_handle.write(main_content)
    file_handle.close()

def remove_numeric_string(text):
    pattern = r'\s*-\s*\d+\s*-\s*'  # 匹配类似于 "- 22 -" 这样的数字字符
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def merge_lines(text):
    lines = text.split("\n")  # 将文本按行分割成列表
    merged_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()  # 去除行首行尾的空白字符

        if i < len(lines) - 1 and not line.endswith("。"):  # 判断当前行是否不以中文句号结尾且不是最后一行
            next_line = lines[i+1].strip()
            merged_line = line + next_line
            merged_lines.append(merged_line)
            i += 2  # 跳过下一行
        else:
            merged_lines.append(line)
            i += 1

    merged_text = "\n".join(merged_lines)
    return merged_text


def merge_lines2(text):
    lines = text.split("\n")  # 将文本按行分割成列表
    merged_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()  # 去除行首行尾的空白字符

        if i < len(lines) - 1 and not re.match(r'^第[零一二三四五六七八九十百千万]+[章节条].*$', line):  # 判断当前行是否不以特定格式开头且不是最后一行
            next_line = lines[i+1].strip()
            merged_line = line + next_line
            merged_lines.append(merged_line)
            i += 2  # 跳过下一行
        else:
            merged_lines.append(line)
            i += 1

    merged_text = "\n".join(merged_lines)
    return merged_text
