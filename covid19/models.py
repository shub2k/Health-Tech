from django.db import models

class Contact(models.Model):
    sno = models.AutoField(primary_key=True)
    name = models.CharField(max_length=300)
    email = models.CharField(max_length=100)
    subject = models.CharField(max_length=100)
    content = models.TextField()
    timestamp  = models.DateTimeField(auto_now_add = True, blank = True)

    def __str__(self):
        return 'Message from ---> ' + self.name + '--->' + self.email 

    
