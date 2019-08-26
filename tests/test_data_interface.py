import os
import json

from pytest import fixture

from data_interface import PostGISQuery

secret_location = os.path.abspath("secret.json")
with open(secret_location) as f:
    secret_password = json.load(f)["password"]

@fixture
def postgis_query():
    return PostGISQuery("osm", "postgres", secret_password)


def test_postgis_query(postgis_query):
    assert postgis_query.con is not None