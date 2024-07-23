import requests

MEDIUM_USER_ID = "cevheri"
MEDIUM_API_URL = f"https://api.medium.com/v1/users/{MEDIUM_USER_ID}/posts"
MEDIUM_TOKEN = "2cd0db9592306382da5142893dfa85f0eb4260b88d70b22d37728b9a12a1521b9"
TITLE = "Gaia dataset and queries with ADQL (Astronomical Data Query Language)"
FILE_NAME = "ADQL.ipynb"


def main():
    headers = {
        "Authorization": f"Bearer {MEDIUM_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Charset": "utf-8"
    }

    data = {
        "title": TITLE,
        "contentFormat": "markdown",
        "content": open(FILE_NAME, "r").read()
    }

    response = requests.post(MEDIUM_API_URL, headers=headers, json=data)

    if response.status_code == 201:
        print("Article published successfully!")
    else:
        print(f"Error: {response.status_code}")
        print(f"Error content: {response.json()}")


if __name__ == "__main__":
    main()
