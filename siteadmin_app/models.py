from django.db import models

# Create your models here.


class SiteAdmin(models.Model):
	name = models.CharField(max_length=10, null=True)
	MANAGEMENT_LOCATIONS_CHOICES = [
		('option1', '구역 1'),
		('option2', '구역 2'),
		('option3', '구역 3'),
	]
	management_locations = models.CharField(max_length=100, choices=MANAGEMENT_LOCATIONS_CHOICES)
	phone = models.CharField(max_length=100)
	created_at = models.DateTimeField(auto_now_add=True, null=True)
	image = models.ImageField(upload_to='siteadmin/', null=False)

