Project Overview
This project demonstrates how to scrape content from Wikipedia, split and embed the content into a Milvus vector database, and use it to answer user queries through a FastAPI interface. The project is divided into four main steps: setting up Milvus, scraping content, embedding the content, and creating a FastAPI service.

## Libraries to be Imported

The following libraries need to be installed to run this project:

- **FastAPI**: Framework for building APIs.
- **Uvicorn**: ASGI server for running the FastAPI application.
- **Requests**: For making HTTP requests to scrape web pages.
- **BeautifulSoup**: Library for parsing HTML and extracting data.
- **LangChain**: Framework for building applications with language models.
- **dotenv**: For loading environment variables from a `.env` file.
- **pymilvus**: Client for interacting with the Milvus vector database.
- **langchain_milvus**: Integration for using Milvus with LangChain.

---------------------------------------------------------------------------------------------------------------------------------------

Step 1: Setting Up Milvus
Milvus is a vector database used to store the embeddings generated from the scraped content. We will use Docker and Docker Compose to set up Milvus.
------------------
1. Install Docker
Follow the instructions based on your operating system to install Docker:

* Docker for Windows
------------------
2. Create Docker Compose File
To set up Milvus, create a docker-compose.yml file in your project directory
--
why docker-compose.yml file?
-
Milvus requires several components (e.g., Milvus server, metadata storage, and data storage) to work together. Manually managing these containers and networking can be cumbersome. A docker-compose.yml file streamlines this process by defining all necessary services, networks, and volumes in a single configuration file, allowing you to manage everything with just one command (docker-compose up).
------------------
3.Run Milvus
Start Milvus using Docker Compose:

--- docker-compose up -d -----

-----------------------------------------------------------------------------------------------------------------------------------------

Step 2: Create the web_scrap.py for Scraping Content.

** This script scrapes Wikipedia content using BeautifulSoup.
-----
Usage
You can use this script by calling the scrape_wikipedia function and passing a Wikipedia URL to it. The function will return the extracted content as a list of strings.

-----------------------------------------------------------------------------------------------------------------------------------------

Step 3: Embedding the Scraped Content into Milvus

** After scraping, the content will be embedded into a vector database (Milvus) using Google Generative AI embeddings.
-----
Usage
1. Scraping and Embedding: To scrape and store the embeddings into Milvus, make a POST request to the /scrape endpoint with a URL.

2. Asking Questions: To query the embedded content, make a GET request to the /ask endpoint with your question.

------------------------------------------------------------------------------------------------------------------------------------------

Step 4: Creating a FastAPI Service
** FastAPI is used to create a REST API for scraping and embedding Wikipedia content, and for querying the Milvus vector database.
----
Endpoints

1.Scraping and Storing with /scrape (POST): This endpoint allows users to scrape a Wikipedia page by providing a URL. The content is processed, split into chunks, and stored as vector embeddings in Milvus. This is crucial for preparing the data to be searched and queried later.

2.Querying with /ask (GET): This endpoint enables users to ask a question based on the previously stored embeddings. It searches the Milvus database for relevant content, and a conversational AI model is used to generate a detailed response based on the found documents, making it a robust way to get answers from the embedded data.

--------------------------------------------------------------------------------------------------------------------------------------------

------------- Conclusion ---------------

 ** You have now set up a Milvus vector database, scraped Wikipedia content, and embedded it using Google Generative AI. You can now use this system to ask questions about the scraped data and get meaningful responses.






