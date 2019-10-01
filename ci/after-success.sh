#!/usr/bin/env bash

set -ex

if [ "${TRAVIS_PULL_REQUEST_BRANCH:-$TRAVIS_BRANCH}" != "master" ] && [ "$TRAVIS_RUST_VERSION" == "stable" ]; then
    REMOTE_URL="$(git config --get remote.origin.url)";
    cd ${TRAVIS_BUILD_DIR}/.. && \
    git clone ${REMOTE_URL} "${TRAVIS_REPO_SLUG}-bench" && \
    cd  "${TRAVIS_REPO_SLUG}-bench" && \
    # Bench master
    git checkout master && \
    cargo bench --bench subword && \
    # Bench pull request
    git checkout ${TRAVIS_COMMIT} && \
    cargo bench --bench subword
fi
