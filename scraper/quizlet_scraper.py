#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quizlet爬虫程序
爬取红楼梦知识竞赛题库的题目和答案
"""

import requests
import json
import time
import re
import random
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import csv
import os

class QuizletScraper:
    def __init__(self):
        """初始化爬虫"""
        self.session = requests.Session()
        self.setup_headers()
        
    def setup_headers(self):
        """设置更真实的请求头"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0'
        ]
        
        self.session.headers.update({
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
        })
        
    def get_page_content(self, url: str) -> str:
        """获取页面内容"""
        try:
            # 添加随机延迟
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"请求失败: {e}")
            return ""
    
    def extract_quizlet_data(self, html_content: str) -> List[Dict[str, str]]:
        """提取Quizlet数据"""
        soup = BeautifulSoup(html_content, 'html.parser')
        flashcards = []
        
        # 尝试多种选择器策略
        selectors = [
            # 策略1: 查找特定的Quizlet类
            '.SetPageTerms-term',
            '.SetPageTerms-termText',
            '[data-testid="term"]',
            '[data-testid="definition"]',
            
            # 策略2: 查找包含题目和答案的容器
            '.SetPageTerms-termContainer',
            '.SetPageTerms-termRow',
            
            # 策略3: 查找闪卡相关元素
            '[class*="card"]',
            '[class*="term"]',
            '[class*="flashcard"]',
            
            # 策略4: 查找文本内容
            'div[class*="text"]',
            'span[class*="text"]',
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                print(f"找到 {len(elements)} 个元素，使用选择器: {selector}")
                flashcards = self.extract_from_elements(elements)
                if flashcards:
                    break
        
        # 如果还是找不到，尝试从页面文本中提取
        if not flashcards:
            flashcards = self.extract_from_page_text(soup)
        
        return flashcards
    
    def extract_from_elements(self, elements) -> List[Dict[str, str]]:
        """从元素中提取题目和答案"""
        flashcards = []
        
        for element in elements:
            try:
                # 尝试提取题目和答案
                text = element.get_text(strip=True)
                if text and len(text) > 5:
                    # 尝试分割题目和答案
                    parts = text.split('\n')
                    if len(parts) >= 2:
                        question = parts[0].strip()
                        answer = parts[1].strip()
                        
                        if question and answer and question != answer:
                            flashcards.append({
                                'question': question,
                                'answer': answer
                            })
            except Exception as e:
                continue
        
        return flashcards
    
    def extract_from_page_text(self, soup) -> List[Dict[str, str]]:
        """从页面文本中提取题目和答案"""
        flashcards = []
        
        try:
            # 获取页面所有文本
            page_text = soup.get_text()
            
            # 使用正则表达式查找可能的题目和答案模式
            # 查找包含问号或冒号的行
            lines = page_text.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or len(line) < 5:
                    continue
                
                # 查找包含问号的行作为题目
                if '？' in line or '?' in line:
                    question = line
                    # 查找下一行作为答案
                    if i + 1 < len(lines):
                        answer = lines[i + 1].strip()
                        if answer and len(answer) > 2:
                            flashcards.append({
                                'question': question,
                                'answer': answer
                            })
                
                # 查找包含冒号的行
                elif '：' in line or ':' in line:
                    parts = line.split('：') if '：' in line else line.split(':')
                    if len(parts) >= 2:
                        question = parts[0].strip()
                        answer = parts[1].strip()
                        if question and answer and len(question) > 2 and len(answer) > 2:
                            flashcards.append({
                                'question': question,
                                'answer': answer
                            })
        
        except Exception as e:
            print(f"从页面文本提取数据时出错: {e}")
        
        return flashcards
    
    def extract_from_api(self, url: str) -> List[Dict[str, str]]:
        """尝试从API获取数据"""
        try:
            # 提取set ID
            set_id_match = re.search(r'/cn/(\d+)/', url)
            if not set_id_match:
                return []
            
            set_id = set_id_match.group(1)
            
            # 尝试访问API端点
            api_url = f"https://quizlet.com/webapi/3.2/sets/{set_id}/terms"
            
            # 添加API特定的请求头
            api_headers = self.session.headers.copy()
            api_headers.update({
                'Accept': 'application/json',
                'X-Requested-With': 'XMLHttpRequest',
            })
            
            response = self.session.get(api_url, headers=api_headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                flashcards = []
                
                for term in data.get('responses', [{}])[0].get('models', {}).get('term', []):
                    flashcards.append({
                        'question': term.get('word', ''),
                        'answer': term.get('definition', '')
                    })
                
                return flashcards
        except Exception as e:
            print(f"API提取失败: {e}")
        
        return []
    
    def scrape_quizlet(self, url: str) -> List[Dict[str, str]]:
        """爬取Quizlet数据"""
        print(f"正在爬取: {url}")
        
        # 首先尝试从API获取数据
        print("尝试从API获取数据...")
        flashcards = self.extract_from_api(url)
        
        if not flashcards:
            # 如果API失败，尝试从HTML提取
            print("API失败，尝试从HTML提取...")
            html_content = self.get_page_content(url)
            if html_content:
                flashcards = self.extract_quizlet_data(html_content)
        
        return flashcards
    
    def save_to_csv(self, flashcards: List[Dict[str, str]], filename: str = "quizlet_data.csv"):
        """保存数据到CSV文件"""
        if not flashcards:
            print("没有数据可保存")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['question', 'answer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for card in flashcards:
                writer.writerow(card)
        
        print(f"数据已保存到 {filename}")
    
    def save_to_json(self, flashcards: List[Dict[str, str]], filename: str = "quizlet_data.json"):
        """保存数据到JSON文件"""
        if not flashcards:
            print("没有数据可保存")
            return
        
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(flashcards, jsonfile, ensure_ascii=False, indent=2)
        
        print(f"数据已保存到 {filename}")

def main():
    """主函数"""
    url = "https://quizlet.com/cn/274823239/%E7%BA%A2%E6%A5%BC%E6%A2%A6%E7%9F%A5%E8%AF%86%E7%AB%9E%E8%B5%9B%E9%A2%98%E5%BA%93-flash-cards/"
    
    scraper = QuizletScraper()
    
    print("🤖 Quizlet爬虫启动")
    print("=" * 50)
    
    # 爬取数据
    flashcards = scraper.scrape_quizlet(url)
    
    if flashcards:
        print(f"✓ 成功爬取到 {len(flashcards)} 个题目")
        
        # 显示前几个题目作为预览
        print("\n📋 数据预览:")
        for i, card in enumerate(flashcards[:5]):
            print(f"{i+1}. 题目: {card['question']}")
            print(f"   答案: {card['answer']}")
            print()
        
        # 保存数据
        scraper.save_to_csv(flashcards)
        scraper.save_to_json(flashcards)
        
    else:
        print("❌ 未能爬取到数据")
        print("可能的原因:")
        print("1. 页面需要登录")
        print("2. 页面结构发生变化")
        print("3. 网络连接问题")
        print("4. 反爬虫机制")
        print("\n建议:")
        print("1. 尝试使用Selenium版本: python quizlet_selenium_scraper.py")
        print("2. 手动访问页面确认是否需要登录")
        print("3. 检查网络连接")

if __name__ == "__main__":
    main() 