#!/usr/bin/env python

import numpy as np
from scipy import special
from scipy.integrate import quad,romberg
from math  import *
import scipy
from scipy.linalg import solve,norm
from scipy import constants
import types
from scipy import constants
from sympy.matrices import *
import scmod as scm

__version__ = '2.0.0'
__name__ = "pyMie"
fd = "yeah!"


class MieSolver(object):
	'''Mie Solver'''

	def __init__(self,freq=None,units="RADPERSEC"):
		self.freq_is_defined = False
		self.radius = None
		self.frequency = freq
		self.units = units
		self.on_off = False
		self.cyl_name = ""
		self.host_name = ""
		self.cyl_n = None
		self.host_n = None
		self.radius=None



	# Set it to True for field calculation and poynting vector
	def field_calculation(self,frequ,on_off=False,):
		self.fieldcalc = on_off
		self.field_freq = frequ
	def te0G(self,y):
		'''
		find the zeros of this function to find the TE0 modes
		works only for mu_h=mu_c=1
		'''

		aa = 4.0j*special.jvp(0,z)*np.log(y)/special.jv(0,y)/np.pi
		bb = self.cyl_n*y**2-4.0j/y/self.host_n
		return aa+bb

	def frequency_range(self,freq,units='RADPERSEC'):

		self.frequency = freq
		if units == "RADPERSEC" or units==None:
			self.frequency = freq
		elif units == "THz":
			self.frequency = self.frequency*1e+12*2.0*np.pi
		elif units == "GHz":
			self.frequency = self.frequency*1e+9*2.0*np.pi
		elif units == "m":
			self.frequency = constants.c*2.0*np.pi/self.frequency
		else:
			raise ValueError("Units are wrong...")

		self.k0 = self.frequency/constants.c
		self.k_cyl = self.k0*self.cyl_n
		self.k_host = self.k0*self.host_n
		self.kR_host = self.k0*self.host_n*self.radius
		self.kR_cyl = self.k0*self.cyl_n*self.radius
		self.freq_is_defined = True



	def units(self,length,frequency):
		pass


	def host(self,hostMaterial):
		# First get epsilon and mu
		self.host_name, self.host_eps, self.host_mu = hostMaterial()
		if np.imag(self.host_eps) != 0.0 or np.imag(self.host_mu) != 0.0:
			raise TypeError("ERROR: Host epsilon and mu must be real numbers.")
		self.host_n = np.sqrt(self.host_eps*self.host_mu)

	def cylinder(self, cylMaterial, radius):
		self.cyl_name, self.cyl_eps, self.cyl_mu = cylMaterial()
		if radius <= 0.0:
			raise ValueError("ERROR: The radius of the cylinder should be a positive number.")

		self.radius = radius
		self.cyl_n = np.sqrt(self.cyl_eps*self.cyl_mu)

	####
	def coeff(self, order, polarization,freq_index=None,denominator=False):
		n = order # for backward compatibility
			# for backward compatibility only

		z1 = self.kR_host
		z2 = self.kR_cyl

		if freq_index>=0:
			z1 = self.kR_host[freq_index]
			z2 = self.kR_cyl[freq_index]
			k0 = self.k0[freq_index]
			kR_cyl = self.kR_cyl[freq_index]
			kR_host = self.kR_host[freq_index]
		else:
			z1 = self.kR_host
			z2 = self.kR_cyl
			k0 = self.k0
			kR_cyl = self.kR_cyl
			kR_host = self.kR_host


				# bessel functions
                J1 = special.jv(order,z1)
                J2 = special.jv(order,z2)
                JD1 = special.jvp(order,z1)
                JD2 = special.jvp(order,z2)
                H1 = special.hankel1(order,z1)
                HD1 = special.h1vp(order,z1)



		mu1 = self.host_mu
		mu2 = self.cyl_mu
		if polarization not in ['E','H','TE','TM']:
			raise ValueError("ERROR: Polarization should be 'E' or 'H'.")

		### POLARIZATION
                if polarization in ['TE','H']:
	    	# scattering coefficienct bn (TE-pol)
			numer = kR_cyl*self.host_mu*J2*JD1-kR_host*self.cyl_mu*J1*JD2
			denom = kR_host*self.cyl_mu*H1*JD2-kR_cyl*self.host_mu*HD1*J2
			#print kR_cyl[0],self.host_mu,J2[0],JD1[0],kR_host[0],J1[0],JD2[0]

			if denominator: return denom
			return numer/denom


                elif polarization in ['TM','E']:
			# scattering coefficienct bn (TM-pol)
			numer = kR_cyl*self.host_mu*J1*JD2-kR_host*self.cyl_mu*J2*JD1
			denom = kR_host*self.cyl_mu*HD1*J2-kR_cyl*self.host_mu*H1*JD2
		        if denominator: return denom
		        return numer/denom

		###############################################

	def coeff_int(self, order, polarization,freq_index=None):
		#if self.frequency == None:
		#	raise Exception("ERROR: Frequency range has not been specified.")
		# for backward compatibility only

		mu1 = self.host_mu
		mu2 = self.cyl_mu
		if freq_index>=0:
			z1 = self.kR_host[freq_index]
			z2 = self.kR_cyl[freq_index]
			k0 = self.k0[freq_index]
			kR_cyl = self.kR_cyl[freq_index]
			kR_host = self.kR_host[freq_index]
		else:
			z1 = self.kR_host
			z2 = self.kR_cyl
			k0 = self.k0
			kR_cyl = self.kR_cyl
			kR_host = self.kR_host


		bn =  self.coeff(order,polarization,freq_index)
		### POLARIZATION
		if polarization in ['TE','H']:
			JD1 = special.jvp(order,z1)
			JD2 = special.jvp(order,z2)
			HD1 = special.h1vp(order,z1)
			numer = HD1*bn+JD1
			denom = JD2
			return numer/denom
		elif polarization in ['TM','E']:
			J2 = special.jv(order,z2)
			J1 = special.jv(order,z1)
			H1 = special.hankel1(order,z1)
			numer = H1*bn+J1
			denom = J2
			return numer/denom
		###############################################



	def mode_efficiencies(self,mode_number,polarization):
		''' Return Qext_n, Qabs_n '''
		'''
		#
		# Calculate the contribution from the n-th mode to Qext
		#
		'''
		#if self.frequency == None:
		#	raise Exception("ERROR: Frequency range has not been specified.")
		en = 1.0
		k0 = self.k0

		kR_host = self.kR_host
		bn = self.coeff(mode_number,polarization)
		ext_term = en*bn
		abs_term = -np.real(bn) - np.real(bn)**2 - np.imag(bn)**2

		return -np.real(bn)/np.abs(kR_host),abs_term/np.abs(kR_host)

	def efficiencies(self,polarization,modes_number=50):
			if len(self.frequency) < 1:
				raise Exception("ERROR: Frequency range has not been specified.")
			''' Return Qsc, Qext, Qabs'''
			num_sc  = 0.0
			num_ext = 0.0
			modes = np.arange(modes_number)

			k0 = self.k0
			kR_host = self.kR_host
			nn = 0.0
			for m in modes:

				if m==0:
					en = 1.0
				else:
					en = 2.0


				bn = self.coeff(m,polarization)
				modeeff = 2.0*self.mode_efficiencies(m,polarization)[0]
				if modeeff.shape != ():
					modeeff = modeeff[0]
				nn = nn + en*modeeff
				sc_term = en*np.conjugate(bn)*bn
				ext_term = en*bn

            		        num_sc += sc_term
            		        num_ext += ext_term

			denom = abs(kR_host)

			# scattering efficiency
			Qsc = 2.0*num_sc/denom
			Qsc = Qsc.real
			# extinction efficiency

  			Qext = -(2.0/np.abs(kR_host))*num_ext.real
			Qabs = Qext - Qsc

  			return Qsc, Qext, Qabs
	####### END FUNCTION #################
	def effective_coated_N0(self,filling,polarization):
		'''
		Calculate effective medium parameters using the
		coated cylinder model
		'''
		R = self.radius
		k = self.k_host
		mu_h = 1.0

		R2 = R/np.sqrt(filling)

		# bessel functions
		J0 = special.jv(0,k*R2)
		H0 = special.hankel1(0,k*R2)
		JD0 = special.jvp(0,k*R2)
		HD0 = special.h1vp(0,k*R2)

		aa0 = self.coeff(0,polarization)
		if polarization in ['H','TE']:
			numer = JD0+HD0*aa0 + 0j
			denom = (J0+ H0*aa0)+ 0j
			term = -2.*mu_h/(k*R2)
			return term*numer/denom + 0j
		else:
			aa0 = self.coeff(0,polarization)
			numer = JD0+HD0*aa0 + 0j
			denom = (J0+ H0*aa0) + 0j
			term = -2*self.host_eps/(k*R2)
			return term*numer/denom + 0j

	def effective_coated_order(self,filling,polarization,order):
		'''
		Calculate effective medium parameters using the
		coated cylinder model
		'''
		R = self.radius
		k = self.k_host
		mu_h = 1.0

		R2 = R/np.sqrt(filling)

		# bessel functions
		J0 = special.jv(order,k*R2)
		H0 = special.hankel1(order,k*R2)
		JD0 = special.jvp(order,k*R2)
		HD0 = special.h1vp(order,k*R2)
		m = order

		#aa0 = self.coeff(1,polarization)
		if polarization in ['H','TE']:
			
			
			
			aa1 = self.coeff(m,'TE')
			denom = JD0+HD0*aa1 + 0j
			numer = (J0+ H0*aa1) + 0j
			sum_ = m*numer/denom
			term = self.host_eps/(k*R2)
			return term*sum_ + 0j
		else:
			
			aa1 = self.coeff(m,'TM')
			denom = JD0+HD0*aa1 + 0j
			numer = (J0+ H0*aa1) + 0j
			sum_ =  numer/denom/m
			term = self.host_mu/(k*R2)
			return term*sum_+ 0j

	def effective_coated_N1(self,filling,polarization,orders=1):
		'''
		Calculate effective medium parameters using the
		coated cylinder model
		'''
		R = self.radius
		k = self.k_host
		mu_h = 1.0

		R2 = R/np.sqrt(filling)

		

		#aa0 = self.coeff(1,polarization)
		sum_ = 0.0+0j
		if polarization in ['H','TE']:
			
			if orders<1: orders = 1
			for m in np.arange(1, orders+1):
				# bessel functions
				J0 = special.jv(m,k*R2)
				H0 = special.hankel1(m,k*R2)
				JD0 = special.jvp(m,k*R2)
				HD0 = special.h1vp(m,k*R2)
				aa1 = self.coeff(m,'TE')
				denom = JD0+HD0*aa1 + 0j
				numer = (J0+ H0*aa1) + 0j
				sum_ = sum_ + m*numer/denom
			term = self.host_eps/(k*R2)
			return term*sum_ + 0j
		else:
			for m in np.arange(1, orders+1):
				J0 = special.jv(m,k*R2)
				H0 = special.hankel1(m,k*R2)
				JD0 = special.jvp(m,k*R2)
				HD0 = special.h1vp(m,k*R2)
				aa1 = self.coeff(m,'TM')
				denom = JD0+HD0*aa1 + 0j
				numer = (J0+ H0*aa1) + 0j
				sum_ = sum_ + m*numer/denom
			term = self.host_mu/(k*R2)
			return term*sum_+ 0j

	def Hfield_mode_outside(self, rho, phi, polarization,mode,freq_index,only_scattered=False):
		'''
		Calculate the magnetic field H outside the cylinder for the a mode
		@arg [double] rho
		@arg [double] phi : polar angle
		@arg [string] polarization: TE or TM
		@arg [int] mode: mode number > 0
		@arg [int] freq_index : self.frequency[freq_index] has to exist.

		@arg [bool] only_scattered : [default] False
		'''
		inc_factor = 1.0
		if only_scattered: inc_factor = 0.0

		coeff = self.coeff(mode, polarization,freq_index)

		factor = 2j**(mode+1)*constants.c/(self.frequency[freq_index]*self.cyl_mu)
		if mode==0: factor = factor/2.0
		if polarization in ['TM' or 'E']:
			 # Mcyl(field,order,rho,phi,component,freq_index)
			 Mcyl_sc  =  self.Mcyl('sc',mode,rho,phi,'all',freq_index) # returns an array
			 Mcyl_inc =  self.Mcyl('inc',mode,rho,phi,'all',freq_index) # returns an array
			 #print Mcyl_inc.shape, Mcyl_sc.shape
			 #return factor*coeff
			 return factor*(inc_factor*Mcyl_inc + coeff*Mcyl_sc)
		else:
			factor = factor/1j
			Ncyl_sc = self.Ncyl("sc",mode,rho,phi,freq_index)
			Ncyl_inc = self.Ncyl("inc", mode, rho, phi, freq_index)
			return factor*(inc_factor*Ncyl_inc + coeff*Ncyl_sc)

	def Hfield_mode_inside(self, rho, phi, polarization,mode,freq_index,only_scattered=False):
		'''
		Calculate the magnetic field H INSIDE the cylinder for the a mode
		@arg [double] rho
		@arg [double] phi : polar angle
		@arg [string] polarization: TE or TM
		@arg [int] mode: mode number > 0
		@arg [int] freq_index : self.frequency[freq_index] has to exist.
		@arg [bool] only_scattered: Return only the scattered field. Set this equal to zero.
		'''

		#if only_scattered: return 0+0j
		coeff = self.coeff_int(mode, polarization,freq_index)

		factor = 2j**(mode+1)*constants.c/(self.frequency[freq_index]*self.cyl_mu)
		if mode==0: factor = factor/2.0
		if polarization in ['TM' or 'E']:
			 # Mcyl(field,order,rho,phi,component,freq_index)
			 Mcyl_int  =  self.Mcyl('int',mode,rho,phi,'all',freq_index) # returns an array

			 #print Mcyl_inc.shape, Mcyl_sc.shape
			 #return factor*coeff
			 return factor*coeff*Mcyl_int
		else:
			factor = factor/1j
			Ncyl_int = self.Ncyl("int",mode,rho,phi,freq_index)
			return factor*coeff*Ncyl_int

	def Efield_mode_outside(self, rho, phi, polarization,mode,freq_index,only_scattered=False):
		'''
		Calculate the electric field E outside the cylinder for the a mode
		@arg [double] rho
		@arg [double] phi : polar angle
		@arg [string] polarization: TE or TM
		@arg [int] mode: mode number > 0
		@arg [int] freq_index : self.frequency[freq_index] has to exist.

		@arg [bool] only_scattered : [default] False
		'''
		inc_factor = 1.0
		if only_scattered: inc_factor = 0.0
		coeff = self.coeff(mode, polarization,freq_index)

		factor = 2j**(mode+1)/self.k_host[freq_index]
		if mode==0: factor = factor/2.
		if polarization in ['TE' or 'H']:

			 # Mcyl(field,order,rho,phi,component,freq_index)
			 Mcyl_sc  =  self.Mcyl('sc',mode,rho,phi,'all',freq_index) # returns an array
			 Mcyl_inc =  self.Mcyl('inc',mode,rho,phi,'all',freq_index) # returns an array
			 #print Mcyl_inc.shape, Mcyl_sc.shape
			 #return factor*coeff
			 return factor*(inc_factor*Mcyl_inc + coeff*Mcyl_sc)
		else:
			factor = factor/1j
			# Ncyl(self,field,order,rho,phi,freq_index)
			Ncyl_sc = self.Ncyl("sc",mode,rho,phi,freq_index)
			Ncyl_inc = self.Ncyl("inc", mode, rho, phi, freq_index)
			return factor*(inc_factor*Ncyl_inc + coeff*Ncyl_sc)

	def Efield_mode_inside(self, rho, phi, polarization,mode,freq_index,only_scattered=False):
		'''
		Calculate the electric field E INSIDE the cylinder for the a mode
		@arg [double] rho
		@arg [double] phi : polar angle
		@arg [string] polarization: TE or TM
		@arg [int] mode: mode number > 0
		@arg [int] freq_index : self.frequency[freq_index] has to exist.

		@arg [bool] only_scattered : [default] False
		'''
		#if only_scattered: return 0+0j
		coeff = self.coeff_int(mode, polarization,freq_index)

		factor = 2j**(mode+1)/self.k_cyl[freq_index]
		if mode==0: factor = factor/2.
		if polarization in ['TE' or 'H']:

			 # Mcyl(field,order,rho,phi,component,freq_index)
			 Mcyl_int  =  self.Mcyl('int',mode,rho,phi,'all',freq_index) # returns an array

			 #print Mcyl_inc.shape, Mcyl_sc.shape
			 #return factor*coeff
			 return factor*coeff*Mcyl_int
		else:
			factor = factor/1j
			# Ncyl(self,field,order,rho,phi,freq_index)
			Ncyl_int = self.Ncyl("int",mode,rho,phi,freq_index)

			return factor*coeff*Ncyl_int

	''' E-field '''
	def Efield(self,freq_index,pol,rho,phi, terms=20,only_scattered=False):
		'''Efield

		parameters
		----------
		freq_index: index correspoding to the frequency.
		pol : polarization
		rho :
		phi : polar angle

		returns
		--------
		array

		'''
		Efield_inside = lambda mode: self.Efield_mode_inside(rho, phi, pol,mode,freq_index,only_scattered)
		Efield_outside = lambda mode: self.Efield_mode_outside(rho, phi, pol,mode,freq_index,only_scattered)
		orders = np.arange(terms)
		def fsum_outside():
			sum_ = 0+0j
			for tt in orders:
				sum_ += Efield_outside(tt)
			return sum_
		def fsum_inside():
			sum_ = 0+0j
			for tt in orders:
				sum_ += Efield_inside(tt)
			return sum_
		#return np.where(rho>self.radius,Efield_outside(0)+Efield_outside(1)+Efield_outside(2),Efield_inside(0)+Efield_inside(1)+Efield_inside(2))
		return np.where(rho>self.radius,fsum_outside(),fsum_inside())

	''' E-field '''
	def Hfield(self,freq_index,pol,rho,phi, terms=20,only_scattered=False):
		'''Efield

		parameters
		----------
		freq_index: index correspoding to the frequency.
		pol : polarization
		rho :
		phi : polar angle

		returns
		--------
		array

		'''
		Hfield_inside = lambda mode: self.Hfield_mode_inside(rho, phi, pol,mode,freq_index,only_scattered)
		Hfield_outside = lambda mode: self.Hfield_mode_outside(rho, phi, pol,mode,freq_index,only_scattered)
		orders = np.arange(terms)
		def fsum_outside():
			sum_ = 0+0j
			for tt in orders:
				sum_ += Hfield_outside(tt)
			return sum_
		def fsum_inside():
			sum_ = 0+0j
			for tt in orders:
				sum_ += Hfield_inside(tt)
			return sum_
		#return np.where(rho>self.radius,Efield_outside(0)+Efield_outside(1)+Efield_outside(2),Efield_inside(0)+Efield_inside(1)+Efield_inside(2))
		return np.where(rho>self.radius,fsum_outside(),fsum_inside())



	'''Cylindical Harmonics'''
	def Ncyl(self,field,order,rho,phi,freq_index):

		n = order
		R = self.radius

		if field=="inc":
			k = self.kR_host[freq_index]/self.radius
			z1 = k*rho
			J1 = special.jv(order,z1)
			return k*J1*np.cos(order*phi)
		elif field=="sc":
			k = self.kR_host[freq_index]/self.radius
			z1 = k*rho
			H1 = special.hankel1(order,z1)
			return k*H1*np.cos(order*phi)
		elif field=="int":
			q = self.kR_cyl[freq_index]/self.radius
			z2 = q*rho
			J2 = special.jv(order,z2)
			return q*J2*np.cos(order*phi)
		else:
			raise Exception("Unknown field in Ncyl")
			return None



	def Mcyl(self,field,order,rho,phi,component,freq_index):

		n = order

		# bessel functions

		if field=="inc":
			k = self.kR_host[freq_index]/self.radius
			z1 = k*rho
			J1 = special.jv(order,z1)
			JD1 =  special.jvp(order,z1)
			m_r   = -order*k*J1 * np.sin(order*phi)/z1
			m_phi =  -k*JD1 * np.cos(order*phi)
		elif field=="sc":
			k = self.kR_host[freq_index]/self.radius
			z1 = k*rho
			H1 = special.hankel1(order,z1)
			HD1 =  special.h1vp(order,z1)
			m_r   = -n*k*H1 * np.sin(order*phi)/z1
			m_phi = -k*HD1 * np.cos(order*phi)
			#print m_r, m_phi
		elif field=="int":
			q = self.kR_cyl[freq_index]/self.radius
			z2 = q*rho
			J2 = special.jv(order,z2)
			JD2 =  special.jvp(order,z2)
			m_r   = -order*q*J2 * np.sin(order*phi)/z2
			m_phi = -q*JD2 * np.cos(order*phi)


		if component in ['r','rho']:
			return m_r
		elif component in ['phi']:
			return m_phi
		else:
			return np.array([m_r,m_phi])






	# CALCULATE THE EFFECTIVE PARAMETERS
	def effective_n1(self,d):
		xsfreq = self.frequency
		phi1 = np.linspace(np.pi/4.0,3.0*np.pi/4.0,500)
		phi2 = np.linspace(-np.pi/4.0,+np.pi/4.0,500)

		mu_yy = np.zeros(len(xsfreq),dtype=np.complex)
		eps_yy = np.zeros(len(xsfreq),dtype=np.complex)
		for ii,freq in enumerate(xsfreq):

		    rBre = lambda pphi: np.real(self.rHy(pphi,d,ii,HorB='B'))
		    rHre = lambda pphi: np.real(self.rHy(pphi,d,ii,HorB='H'))
		    rBim = lambda pphi: np.imag(self.rHy(pphi,d,ii,HorB='B'))
		    rHim = lambda pphi: np.imag(self.rHy(pphi,d,ii,HorB='H'))

		    rDre = lambda pphi: np.real(self.rEy(pphi,d,ii,EorD='D'))
		    rDim = lambda pphi: np.imag(self.rEy(pphi,d,ii,EorD='D'))
		    rEre = lambda pphi: np.real(self.rEy(pphi,d,ii,EorD='E'))
		    rEim = lambda pphi: np.imag(self.rEy(pphi,d,ii,EorD='E'))


		    fH = np.zeros(phi1.shape,dtype=complex)
		    fB = np.zeros(phi2.shape,dtype=complex)
		    fE = np.zeros(phi1.shape,dtype=complex)
		    fD = np.zeros(phi2.shape,dtype=complex)

		    I_Dre,err5 = quad(rDre,phi1[0],phi1[-1])
		    I_Dim,err6 = quad(rDim,phi1[0],phi1[-1])
		    I_Ere,err7 = quad(rEre,phi2[0],phi2[-1])
		    I_Eim,err8 = quad(rEim,phi2[0],phi2[-1])


		    I_Bre,err1 = quad(rBre,phi1[0],phi1[-1])
		    I_Bim,err2 = quad(rBim,phi1[0],phi1[-1])
		    I_Hre,err3 = quad(rHre,phi2[0],phi2[-1])
		    I_Him,err4 = quad(rHim,phi2[0],phi2[-1])

		    I_B = I_Bre+1.0j*I_Bim
		    I_H = I_Hre+1.0j*I_Him
		    I_D = I_Dre+1.0j*I_Dim
		    I_E = I_Ere+1.0j*I_Eim

		    mu_yy[ii] = I_B*I_H.conjugate()/np.abs(I_H)**2
		    eps_yy[ii] = I_D*I_E.conjugate()/np.abs(I_E)**2
		return mu_yy,eps_yy
	#
	def rHy(self,phi,spacing,freq_index,HorB='H'):
		'''For effective medium calculations for the n=1 mode and TM polarization only'''
		if HorB == 'H':
			rho_ = 0.5*spacing/np.cos(phi)
		elif HorB == 'B':
			rho_ = 0.5*spacing/np.sin(phi)

		field = self.Hfield_mode_outside(rho_, phi, 'TM',1,freq_index)#self.Hfield("TM",rho_,phi,terms=1,n1=True)


		Hy = field[0]*np.sin(phi)+field[1]*np.cos(phi)
		return rho_*Hy

	# calculate rho*Ey (for effective medium use ONLY)
	# and TE polarization ???
	def rEy(self,phi,d,freq_index,EorD='E'):

	    if EorD == "E":
	        rho_ = d/(2.0*np.cos(phi))
	    elif EorD == "D":
	        rho_ = self.host_eps*d/(2.0*np.sin(phi)) # the dielectric constant is included here

	    # calculate Hrho, Hphi
	    # Efield(pol,k,q,R,eps1,eps2,mu1,mu2,rho,phi,component,rho_greater)
	    field = self.Efield_mode_outside(rho_, phi, 'TE',1,freq_index)#self.Hfield("TM",rho_,phi,terms=1,n1=True)

	    Ey = field[0]*np.sin(phi)+field[1]*np.cos(phi)
	    #Ex = field[0]*np.cos(phi) - field[1]*np.sin(phi)
	    return rho_*Ey

	# Maxwell-Garnett TE
	def maxwellGarnettTE(self,fr):
	    a = 1. + fr
	    b = 1. - fr

	    epscyl = self.cyl_eps
	    epshost = self.host_eps

	    C = a*epscyl + b*epshost
	    D = b*epscyl + a*epshost
	    E = C/D
	    return epshost*E

	# Maxwell-Garnett TM
	def maxwellGarnettTM(self,fr):
		#print epscyl
		epscyl = self.cyl_eps
		epshost = self.host_eps
		return fr*epscyl + (1.0-fr)*epshost

	def effective(self,polarization,spacing):
		'''Calculates the effective pamateres for the n=0 mode
		based of the field-averaging method.

		'''
		if spacing < 2.0*self.radius:
			raise Exception("ERROR: Lattice spacing MUST BE greater than 2*R.")
    	#impedances
		eta_h = np.sqrt(self.host_mu/self.host_eps)
		eta_c = np.sqrt(self.cyl_mu/self.cyl_eps)
		R = self.radius
		z1 = self.kR_host
		z2 = self.kR_cyl
		k = self.k_host
		q = self.k_cyl
		d = spacing # lattice spacing
		tt = d/np.sqrt(np.pi)
	    # bessel functions
		J0 = special.jv(0,k*d/2.0)
		J1kR = special.jv(1,k*R)
		J1qR = special.jv(1,q*R)
		J1dpi = special.jv(1,k*tt)
		H0 = special.hankel1(0,k*d/2.0)
		H1kR = special.hankel1(1,k*R)
		H1dpi = special.hankel1(1,k*tt)

		epsC = self.cyl_eps
		epsH = self.host_eps

		# Coeffi
		aa0 = self.coeff(0,polarization)
		cc0 = self.coeff_int(0,polarization)

		if polarization in ["TE","H"]:
			# CALCULATE mu_{eff}
			A = 2.0*np.pi*eta_h/d**2
			D = (J0 + aa0*H0) #/eta_h # E_z
			B = cc0*R*J1qR*self.cyl_mu/(q*eta_c)

		    # Here, i have assumed that mu_h = 1 (?)
			#C = ((tt*J1dpi-R*J1kR)+aa0*(tt*H1dpi-R*H1kR))/(k*eta_h)

			C1 = tt*J1dpi-R*J1kR
			C2 = tt*H1dpi-R*H1kR
			term1 = self.host_mu/(k*eta_h)
			term2 = term1*aa0
			C = term1*C1 + term2*C2
			eff_0 = A*(B+C)/D # resonant mu
			denom = D*eta_h
		else:
	        # CALCULATE epsilon_{eff} for the TM0 mode
			A = 2.0*pi/d**2
			B = cc0*R*J1qR*self.cyl_eps/q
			D = (J0 + aa0*H0) # denominator

			C = ((tt*J1dpi-R*J1kR)*epsH+epsH*aa0*(tt*H1dpi-R*H1kR))/k
	        #denom = D

	        # epsC*B = epsH*C = <Dz>
	        # D = <Ez>
			eff_0 = A*(B+C)/D # resonant epsilon

		return eff_0

	####
	def coeff_func(self, p,  frequency, order, filling, polarization):
		n = order # for backward compatibility
			# for backward compatibility only
		eps_eff, mu_eff = p
		ind = 1
		if eps_eff.real < 0 and mu_eff.real<0: ind = -1
		n_eff = ind* np.sqrt(eps_eff*mu_eff)
		R2 = self.radius * np.sqrt(filling)
		denominator = False

		z1 = self.host_n*2*np.pi*frequency*R2/constants.c
		z2 = n_eff*2*np.pi*frequency*R2/constants.c
		freq_index = 0
		
		
		#z2 = self.kR_cyl
		#k0 = self.k0
		kR_cyl = z2
		kR_host = z1


				# bessel functions
                J1 = special.jv(order,z1)
                J2 = special.jv(order,z2)
                JD1 = special.jvp(order,z1)
                JD2 = special.jvp(order,z2)
                H1 = special.hankel1(order,z1)
                HD1 = special.h1vp(order,z1)



		mu1 = self.host_mu
		mu2 = mu_eff
		if polarization not in ['E','H','TE','TM']:
			raise ValueError("ERROR: Polarization should be 'E' or 'H'.")

		### POLARIZATION
                if polarization in ['TE','H']:
	    	# scattering coefficienct bn (TE-pol)
			numer = kR_cyl*self.host_mu*J2*JD1-kR_host*self.cyl_mu*J1*JD2
			denom = kR_host*self.cyl_mu*H1*JD2-kR_cyl*self.host_mu*HD1*J2
			#print kR_cyl[0],self.host_mu,J2[0],JD1[0],kR_host[0],J1[0],JD2[0]

			if denominator: return denom
			return numer/denom


                elif polarization in ['TM','E']:
			# scattering coefficienct bn (TM-pol)
			numer = kR_cyl*self.host_mu*J1*JD2-kR_host*self.cyl_mu*J2*JD1
			denom = kR_host*self.cyl_mu*HD1*J2-kR_cyl*self.host_mu*H1*JD2
		        if denominator: return denom
		        aa = numer/denom
		        return aa #(np.real(aa), np.imag(aa))





# Class: Material
class Material(object):
	""" Material Base Class.  """
	def __init__(self,name="",eps=1.0,mu=1.0,frequency=None,freqUnits=None):
		self.epsilon = eps
		self.mu = mu
		self.frequency = frequency
		self.name = name
		self.drude_plasmaFreq = 0.0
		self.drude_gamma = 0.0
		self.lorentz_wto = 0.0
		self.lorentz_gammato = 0.0
		self.lorentz_strength = 0.0
		self.lorentz_epsinf = 0.0
		self.lorentz_eps0 = 0.0

	def frequency_range(self,freq,units='RADPERSEC'):
		self.frequency = freq
		if units == "RADPERSEC":
			self.frequency = freq
		elif units == "THz":
			self.frequency = self.frequency*1e+12*2.0*np.pi
		elif units == "GHz":
			self.frequency = self.frequency*1e+9*2.0*np.pi
		elif units == "m":
			self.frequency = constants.c*2.0*np.pi/self.frequency
		else:
			raise ValueError("Units are wrong...")

	def __call__(self):
		return self.name, self.epsilon, self.mu

	def set_epsilon(self,eps=1.0):
		if (self.eps  == None) or (eps == None):
			raise ValueError("ERROR: Please specify epsilon.")
		return self.epsilon

	def get_epsilon(self,frequency=None,units="RADPERSEC"):
		'''
		BUGGY: Rewrite it.
		'''
		self.epsilon = 0.0
		self.frequency_range(frequency,units=units)
		self.addDrude(self.drude_plasmaFreq,self.drude_plasmaFreq)
		self.addLorentz(self.lorentz_wto,self.lorentz_gammato,self.lorentz_strength)
		return self.epsilon

	def set_mu(self,mmu=1.0):
		if mu == None:
			raise ValueError("ERROR: Please specify mu.")
		return self.mu


	""" Material Models """
	def addDrude(self,plasma_freq,gamma):
		'''
		@TODO : Add tests for the values of plasma_freq and gamma_freq
		'''
		self.drude_plasmaFreq = plasma_freq
		self.drude_gamma = gamma
		if self.frequency == None:
			raise Exception("ERROR: Specify the frequency first.")
		drude_term = plasma_freq**2/(self.frequency**2 + 1.0j*self.frequency*gamma)
		self.epsilon = self.epsilon - drude_term

	def addLorentz(self,res_freq, gamma, strength):
		''' Add Lorentz susceptibility.
		strength = eps_0 - eps_inf
		'''
		self.lorentz_epsinf = self.epsilon
		self.lorentz_eps0 = self.epsilon + strength
		self.lorentz_wto = res_freq
		self.lorentz_gammato = gamma
		self.lorentz_strength = strength
		if not isinstance(self.frequency,np.ndarray):
			raise Exception("ERROR: Specify the frequency first.")
		lorentz_term = strength*res_freq**2/(res_freq**2-self.frequency**2-1.0j*self.frequency*gamma)
		#print lorentz_term
		#print self.epsilon
		self.epsilon = self.epsilon + lorentz_term

	def Lorentz_refrindex(self,freq,units='THz'):
		'''
		Returns the refractive index for a lorentzian term
		@arg [ndarray] freq: frequencies in THz
		@return ndarray of frequency in THz
		'''
		res_freq = self.lorentz_wto*1e-12
		gamma= self.lorentz_gammato*1e-12
		eps0 = self.lorentz_eps0
		epsinf = self.lorentz_epsinf
		#w = 2.0*np.pi*freq
		a = (eps0-epsinf)*res_freq**2
		return  np.sqrt(epsinf+a/(res_freq**2-(2*np.pi*freq)**2-1.0j*(2*np.pi*freq)*gamma))


	def Lorentz(self,freq):
		'''
		Returns a Lorentzian
		@arg [ndarray] freq: frequencies in Hz
		@return ndarray of frequency in Hz
		'''
		res_freq = self.lorentz_wto
		gamma= self.lorentz_gammato
		eps0 = self.lorentz_eps0
		epsinf = self.lorentz_epsinf
		w = 2.0*np.pi*freq
		a = (eps0-epsinf)*res_freq**2
		bb =  epsinf+a/(res_freq**2-w**2-1.0j*w*gamma)
		return bb








#######################################################
############################################################
## EFFECTIVE MEDIUM ###################################
############################################################
# CALCULATE THE EFFECTIVE PARAMETERS


# calculate rho*Hy (for effective medium use ONLY)
# and TM polarization
def rHy_(phi,term,k,q,R,d,epsH,epsC,muH,muC):

    if term == "H":
        rho = d/(2.0*np.cos(phi))
    elif term == "B":
        rho = d/(2.0*np.sin(phi))

    # calculate Hrho, Hphi
    if rho>R:
    	#Hrho = Hfield('TM',k,q,R,epsH,epsC,muH,muC,rho,phi,'r',terms=1)
    	#Hphi = Hfield('TM',k,q,R,epsH,epsC,muH,muC,rho,phi,'phi',terms=1)
    	Hrho = Hfield_n1('TM',k,q,R,epsH,epsC,muH,muC,rho,phi,'r')
    	Hphi = Hfield_n1('TM',k,q,R,epsH,epsC,muH,muC,rho,phi,'phi')

    else:
    	Hrho = 0.0+0.0j
    	Hphi = 0.0+0.0j
    	#print "r<R"
    # calculate Hy
    Hy = Hrho*np.sin(phi)+Hphi*np.cos(phi)

    return rho*Hy

# Maxwell-Garnett TE
def maxwellGarnettTE(fr, epscyl, epshost):
    a = 1. + fr
    b = 1. - fr

    C = a*epscyl + b*epshost
    D = b*epscyl + a*epshost
    E = C/D
    return epshost*E

# Maxwell-Garnett TM
def maxwellGarnettTM(fr,epscyl,epshost):
	#print epscyl
	return fr*epscyl + (1.0-fr)*epshost
