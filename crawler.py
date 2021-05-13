from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from utils.utils import load_config

# Load configuration
config = load_config()

# Starts webdriver
driver = webdriver.Chrome(ChromeDriverManager().install())

# Loads urls to get from config file
urls_dict = config["crawler"]["urls_to_get"]

errors = []
results = []

# For each bairro get the number of pages set on configuration
for bairro, url in urls_dict.items():
    current_url = url
    driver.get(url)
    sleep(2)
    actions = ActionChains(driver)

    try:
        driver.find_element_by_class_name("cookie-notifier__cta").click()
    except:
        print("No cookies!")

    for i in tqdm(range(config["crawler"]["pages_to_get"]), desc=bairro):
        sleep(5)
        main_div = driver.find_element_by_class_name("results-main__panel")
        properties = main_div.find_elements_by_class_name("js-property-card")
        paginator = driver.find_element_by_class_name("js-results-pagination")
        next_page = paginator.find_element_by_xpath("//a[@title='Próxima página']")

        for i, apartment in enumerate(properties):
            url = apartment.find_element_by_class_name("js-card-title").get_attribute(
                "href"
            )
            apto_id = url.split("id-")[-1][:-1]
            header = apartment.find_element_by_class_name("property-card__title").text
            address = apartment.find_element_by_class_name(
                "property-card__address"
            ).text
            area = apartment.find_element_by_class_name(
                "js-property-card-detail-area"
            ).text
            rooms = apartment.find_element_by_class_name(
                "js-property-detail-rooms"
            ).text
            bathrooms = apartment.find_element_by_class_name(
                "js-property-detail-bathroom"
            ).text
            garages = apartment.find_element_by_class_name(
                "js-property-detail-garages"
            ).text
            try:
                amenities = apartment.find_element_by_class_name(
                    "property-card__amenities"
                ).text
            except:
                amenities = None
                errors.append(url)
            price = apartment.find_element_by_class_name("js-property-card-prices").text
            try:
                condo = apartment.find_element_by_class_name("js-condo-price").text
            except:
                condo = None
                errors.append(url)
            crawler = bairro
            crawled_at = datetime.now().strftime("%Y-%m-%d %H:%M")
            results.append(
                {
                    "id": apto_id,
                    "url": url,
                    "header": header,
                    "address": address,
                    "area": area,
                    "rooms": rooms,
                    "bathrooms": bathrooms,
                    "garages": garages,
                    "amenities": amenities,
                    "price": price,
                    "condo": condo,
                    "crawler": crawler,
                    "crawled_at": crawled_at,
                }
            )
        try:
            next_page.click()
        except:
            print("Next page not clickable")
            break

# Saving results fo CSV
print(f"Saving to {config['paths']['raw']}")
pd.DataFrame(results).to_csv(config["paths"]["raw"], index=False)
driver.close()
