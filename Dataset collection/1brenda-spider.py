import re

import pandas as pd
from lxml import etree
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait as Wait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import random

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm


class BrendaSpider:
    def __init__(self):
        profile = webdriver.FirefoxOptions()
        profile.set_preference("browser.startup.homepage", "about:blank")
        profile.set_preference("startup.homepage_welcome_url", "about:blank")
        profile.set_preference("startup.homepage_welcome_url.additional", "about:blank")

        desired_capabilities = DesiredCapabilities.FIREFOX  # 修改页面加载策略
        desired_capabilities["pageLoadStrategy"] = "none"  # 注释这两行会导致最后输出结果的延迟，即等待页面加载完成再输出

        self.driver = webdriver.Firefox()

        self.driver.get("https://www.brenda-enzymes.org/index.php")

    # the action of clicking
    def click(self, selector):
        self.driver.find_element(By.CSS_SELECTOR, selector).click()

    def wait(self, selector):
        Wait(self.driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))

    def wait_disappear(self, selector):
        Wait(self.driver, 30).until(EC.invisibility_of_element_located((By.CSS_SELECTOR, selector)))

    def send_keys(self, selector, keys):
        self.driver.find_element(By.CSS_SELECTOR, selector).clear()
        self.driver.find_element(By.CSS_SELECTOR, selector).send_keys(keys)

    def search(self, ec_number):
        try:
            tgt_url = f"https://www.brenda-enzymes.org/enzyme.php?ecno={ec_number}#kcat/KM%20VALUE%20[1/mMs%3Csup%3E-1%3C/sup%3E]"
            self.driver.get(tgt_url)
        except TimeoutException:
            return False

        wrapper = ".flat-wrapper"
        unfold_btn = "#tab305_se"
        table_elem = "#tab305_head"

        self.wait(wrapper)
        if 'Error 404 - Enzyme not found!' in self.driver.page_source:
            return False

        try:
            self.wait(table_elem)

            try:
                self.click(unfold_btn)
            except NoSuchElementException:
                ...
            finally:
                return True

        except TimeoutException:
            return False

    def analysis_page(self):
        html = self.driver.page_source
        tree = etree.HTML(html)

        values = tree.xpath('//*[@id="tab305"]/div[contains(@class, "row")]//div[1]//span/text()')
        substrate = tree.xpath('//*[@id="tab305"]/div[contains(@class, "row")]//div[2]//span/text()')

        organism_rows = tree.xpath('//*[@id="tab305"]/div[contains(@class, "row")]//div[3]//span')
        organism = [x.xpath('string(.)') for x in organism_rows]

        uniprot_rows = tree.xpath('//*[@id="tab305"]/div[contains(@class, "row")]//div[4]//span')
        uniprot = [x.xpath('string(.)') for x in uniprot_rows]

        commentary = tree.xpath('//*[@id="tab305"]/div[contains(@class, "row")]//div[5]//span/text()')

        n_item = min([
            len(organism), len(substrate), len(organism), len(uniprot), len(commentary)
        ])

        return {
            'values': values[:n_item],
            'substrate': substrate[:n_item],
            'organism': organism[:n_item],
            'uniprot': uniprot[:n_item],
            'commentary': commentary[:n_item]
        }

    def run(self, ec_number):
        flag = self.search(ec_number)
        if flag:
            return pd.DataFrame(self.analysis_page())
        return pd.DataFrame()


if __name__ == "__main__":
    import os

    root = r"F:\SecondStudy\brenda_ratio2"
    tgt = "./kotori_spider"

    files = os.listdir(root)
    existed_files = [os.path.splitext(x)[0] for x in os.listdir(tgt)]

    demo = BrendaSpider()
    for file in files:
        ecnumber, _ = os.path.splitext(file)

        if ecnumber in existed_files:
            continue

        try:
            out = pd.DataFrame(demo.run(ecnumber[2:]))
            if not out.empty:
                out = out.applymap(lambda x: x.strip()).applymap(lambda x: None if x == "-" else x)
                out.insert(0, 'ec_number', ecnumber)
                out.to_csv(f'./kotori_spider/{ecnumber}.txt', sep="\t", index=False)
        except:
            ...
        finally:
            demo.driver.get('about:blank')
            time.sleep(0.5)
    # print(pd.DataFrame(ans_))
