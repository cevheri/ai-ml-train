import requests


def main():
    MEDIUM_USER_ID = "xxx"
    MEDIUM_API_URL = f"https://api.medium.com/v1/users/{MEDIUM_USER_ID}/posts"
    MEDIUM_TOKEN = "xxx"

    headers = {
        "Authorization": f"Bearer {MEDIUM_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Charset": "utf-8"
    }

    data = {
        "title": "Astronomical Data Analysis with Python Using Astropy and Astroquery",
        "contentFormat": "markdown",
        "content": open("intro-to-astro.md", "r").read()
    }

    response = requests.post(MEDIUM_API_URL, headers=headers, json=data)

    if response.status_code == 201:
        print("Article published successfully!")
    else:
        print(f"Error: {response.status_code}")
        print(f"Error content: {response.json()}")


if __name__ == "__main__":
    main()
