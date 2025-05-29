# :earth_americas: Tidee Application

A SMDA application that generates velocity trends using sonic log and checkshot data from cloud.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gdp-dashboard-template.streamlit.app/)

### How to run it on your own machine. Please do not hesitate to contact @apenh in case you need any clarification.

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```
2. Add a folder called smda_password on the main .\onecheckshot_inversion_app page with a config.yaml, els_api.yaml, and smda_api.yaml file containing user and password for smda and els api's.

2. Go to .\onecheckshot_inversion_app in powershell and run:

   ```
   streamlit run _1_Introduction_tidee.py
   ```

3. The App page will open.

4. Once you check everything is working as expected locally, you can go to docker-compose.yml and use "Compose Up". Please verify that the image is working correctly in Docker Desktop and that you are logged in on the right azure subscription (use az login for this purpose). Then, one can go to the docker section in VSCode and push the image to Azure registry.

5. Another command that can work is the following:
az acr build --registry tidee --image onecheckshot_inversion_app:latest --file Dockerfile .
