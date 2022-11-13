import unittest
import requests


class MyTestCase(unittest.TestCase):
    def test_something(self):

        resp = requests.post("http://127.0.0.1:5000/predict",
                             files={"file": open('cat.jpg', 'rb')})
        print(resp.json())
        self.assertEqual("{'class_id': 'n02124075', 'class_name': 'Egyptian_cat'}", resp.json())


if __name__ == '__main__':
    unittest.main()
