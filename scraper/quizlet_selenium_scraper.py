#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quizlet Selenium爬虫程序
使用Selenium处理动态加载的内容，爬取红楼梦知识竞赛题库
"""

import time
import json
import csv
import re
import random
from typing import List, Dict, Any
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains

class QuizletSeleniumScraper:
    def __init__(self, headless: bool = True):
        """初始化Selenium爬虫"""
        self.driver = None
        self.headless = headless
        self.setup_driver()
    
    def setup_driver(self):
        """设置Chrome驱动"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            # 执行脚本来隐藏webdriver特征
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            print("✓ Chrome驱动初始化成功")
        except Exception as e:
            print(f"❌ Chrome驱动初始化失败: {e}")
            print("请确保已安装Chrome浏览器和ChromeDriver")
    
    def wait_for_element(self, by, value, timeout=10):
        """等待元素出现"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except TimeoutException:
            return None
    
    def scroll_page(self):
        """滚动页面以加载更多内容"""
        try:
            # 滚动到页面底部
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # 滚动到页面顶部
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)
            
            # 随机滚动
            for i in range(3):
                scroll_height = random.randint(300, 800)
                self.driver.execute_script(f"window.scrollTo(0, {scroll_height});")
                time.sleep(random.uniform(0.5, 1.5))
        
        except Exception as e:
            print(f"滚动页面时出错: {e}")
    
    def extract_flashcards(self) -> List[Dict[str, str]]:
        """提取闪卡数据"""
        flashcards = []
        
        try:
            # 等待页面加载
            time.sleep(5)
            
            # 滚动页面以加载更多内容
            self.scroll_page()
            
            # 尝试多种选择器策略
            selectors = [
                # 策略1: Quizlet特定的选择器
                '[data-testid="flashcard"]',
                '.SetPageTerms-term',
                '.SetPageTerms-termText',
                '[data-testid="term"]',
                '[data-testid="definition"]',
                
                # 策略2: 通用的闪卡选择器
                '[class*="card"]',
                '[class*="term"]',
                '[class*="flashcard"]',
                
                # 策略3: 文本内容选择器
                'div[class*="text"]',
                'span[class*="text"]',
                'p[class*="text"]',
                
                # 策略4: 更通用的选择器
                'div[class*="content"]',
                'div[class*="item"]',
                'li[class*="item"]',
            ]
            
            for selector in selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        print(f"找到 {len(elements)} 个元素，使用选择器: {selector}")
                        flashcards = self.extract_from_elements(elements)
                        if flashcards:
                            break
                except Exception as e:
                    print(f"选择器 {selector} 失败: {e}")
                    continue
            
            # 如果还是找不到，尝试从页面文本中提取
            if not flashcards:
                flashcards = self.extract_from_page_text()
        
        except Exception as e:
            print(f"提取闪卡数据时出错: {e}")
        
        return flashcards
    
    def extract_from_elements(self, elements) -> List[Dict[str, str]]:
        """从元素中提取题目和答案"""
        flashcards = []
        
        for element in elements:
            try:
                # 获取元素文本
                text = element.text.strip()
                if not text or len(text) < 5:
                    continue
                
                # 尝试分割题目和答案
                lines = text.split('\n')
                if len(lines) >= 2:
                    question = lines[0].strip()
                    answer = lines[1].strip()
                    
                    if question and answer and question != answer and len(question) > 2 and len(answer) > 2:
                        flashcards.append({
                            'question': question,
                            'answer': answer
                        })
                
                # 如果只有一行，尝试查找子元素
                elif len(lines) == 1:
                    # 查找子元素
                    child_elements = element.find_elements(By.CSS_SELECTOR, '*')
                    if len(child_elements) >= 2:
                        question = child_elements[0].text.strip()
                        answer = child_elements[1].text.strip()
                        
                        if question and answer and question != answer:
                            flashcards.append({
                                'question': question,
                                'answer': answer
                            })
            
            except Exception as e:
                print(f"提取单个元素时出错: {e}")
                continue
        
        return flashcards
    
    def extract_from_page_text(self) -> List[Dict[str, str]]:
        """从页面文本中提取题目和答案"""
        flashcards = []
        
        try:
            # 获取页面所有文本
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            
            # 使用正则表达式查找可能的题目和答案
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
    
    def scrape_quizlet(self, url: str) -> List[Dict[str, str]]:
        """爬取Quizlet数据"""
        if not self.driver:
            print("❌ 驱动未初始化")
            return []
        
        try:
            print(f"正在访问: {url}")
            self.driver.get(url)
            
            # 等待页面加载
            time.sleep(8)
            
            # 尝试点击"显示所有"按钮（如果存在）
            try:
                show_all_selectors = [
                    "//button[contains(text(), '显示所有')]",
                    "//button[contains(text(), 'Show all')]",
                    "//button[contains(text(), 'View all')]",
                    "//a[contains(text(), '显示所有')]",
                    "//a[contains(text(), 'Show all')]",
                ]
                
                for selector in show_all_selectors:
                    try:
                        show_all_button = self.driver.find_element(By.XPATH, selector)
                        show_all_button.click()
                        print("✓ 点击了显示所有按钮")
                        time.sleep(3)
                        break
                    except NoSuchElementException:
                        continue
                        
            except Exception as e:
                print(f"点击显示所有按钮时出错: {e}")
            
            # 尝试点击"加载更多"按钮
            try:
                load_more_selectors = [
                    "//button[contains(text(), '加载更多')]",
                    "//button[contains(text(), 'Load more')]",
                    "//button[contains(text(), '更多')]",
                ]
                
                for selector in load_more_selectors:
                    try:
                        load_more_button = self.driver.find_element(By.XPATH, selector)
                        load_more_button.click()
                        print("✓ 点击了加载更多按钮")
                        time.sleep(2)
                    except NoSuchElementException:
                        continue
                        
            except Exception as e:
                print(f"点击加载更多按钮时出错: {e}")
            
            # 提取数据
            flashcards = self.extract_flashcards()
            
            return flashcards
        
        except Exception as e:
            print(f"爬取过程中出错: {e}")
            return []
    
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
    
    def close(self):
        """关闭浏览器"""
        if self.driver:
            self.driver.quit()
            print("✓ 浏览器已关闭")

def main():
    """主函数"""
    url = "https://quizlet.com/cn/274823239/%E7%BA%A2%E6%A5%BC%E6%A2%A6%E7%9F%A5%E8%AF%86%E7%AB%9E%E8%B5%9B%E9%A2%98%E5%BA%93-flash-cards/"
    
    scraper = QuizletSeleniumScraper(headless=False)  # 设置为False可以看到浏览器操作
    
    print("🤖 Quizlet Selenium爬虫启动")
    print("=" * 50)
    
    try:
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
            print("1. 手动访问页面确认是否需要登录")
            print("2. 检查网络连接")
            print("3. 尝试使用不同的浏览器设置")
    
    finally:
        # 关闭浏览器
        scraper.close()

if __name__ == "__main__":
    main() 