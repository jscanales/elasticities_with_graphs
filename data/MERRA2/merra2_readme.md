
# Steps to download files from MERRA2:
Before being able to create a database, the files from MERRA-2 containing wind speeds and temperature must be downloaded manually.
1) Register an account in NASA
2) Write the credentials in the file 'config.py'
3) Create a new subset file for MERRA2 describing the start and end dates as well as the desired coordinates using the file 'expanding_merra.py'
4) Update the request header in the file 'config.py'
5) Run the file 'merra2.py' and wait for all files to be downloaded

After that, the database can be built.

# Steps to update the request header:
To update the cookie for MERRA-2 requests, you'll need to generate a new authentication session with Earthdata. At least the cookie field must be updated, but if the 'merra2.py' file doesn't run after updating the cookie, then continue updating fields. 
For any field, follow these steps:
1) Log in: Open one of the links from the subset that you want to download and log in with your credentials
2) Get the cookie:
    * Once logged in, use a browser to open the Developer Tools (press F12 or Ctrl+Shift+I in most browsers).
    * Go to the Network tab, and perform any action that sends a request.
    * Look for the request. Select it and inspect the request headers.
    * You should see a Cookie header that contains the relevant session information.
3) Copy and update the cookie in your script: In your Nasa class within the 'config.py' file, replace the current Cookie value in the header with the new one you copied from the browser's developer tools.