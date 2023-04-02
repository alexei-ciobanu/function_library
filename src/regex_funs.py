import re

re_last_number = re.compile(r'(\d+)(?!.*\d)')
re_last_digits = re.compile(r'(\d*)$')

get_last_number_from_string = lambda x: int(re_last_number.findall(x)[0])
get_last_digits_from_string = lambda x: int(re_last_number.findall(x)[0])

re_m_fun = lambda x: re.compile(rf"{x}\[(.*?)\]")
re_m_sqrt = re_m_fun('Sqrt')
re_m_erfi = re_m_fun('Erfi')
re_m_pow = re.compile(r'\^')
re_m_pi = re.compile(r"(?<![\w])Pi(?![\w])")
re_m_i = re.compile(r"(?<![\w])I(?![\w])")
re_m_exp = re.compile(r"(?<![\w])E\^(?![\w])")